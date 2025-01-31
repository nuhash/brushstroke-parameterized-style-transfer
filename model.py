import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    import tensorflow as tf
    from tensorflow.core.protobuf import config_pb2

import os
import numpy as np
from PIL import Image
from tqdm import trange
import skimage.transform
import networks
import ops
import utils

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def stylize(content_img,
            style_img,
            # Brushstroke optimizer params
            resolution=512,
            num_strokes=5000,
            num_steps=100,
            S=10,
            K=20,
            canvas_color='gray',
            width_scale=0.1,
            length_scale=1.1,
            content_weight=1.0,
            style_weight=3.0,
            tv_weight=0.008,
            curviture_weight=4.0,
            # Pixel optimizer params
            pixel_resolution=1024,
            num_steps_pixel=2000
            ):

    stroke_optim = BrushstrokeOptimizer(content_img,
                                        style_img,
                                        resolution=resolution,
                                        num_strokes=num_strokes,
                                        num_steps=num_steps,
                                        S=S,
                                        K=K,
                                        canvas_color=canvas_color,
                                        width_scale=width_scale,
                                        length_scale=length_scale,
                                        content_weight=content_weight,
                                        style_weight=style_weight,
                                        tv_weight=tv_weight,
                                        curviture_weight=curviture_weight)
    print('Stroke optimization:')
    canvas = stroke_optim.optimize()

    pixel_optim = PixelOptimizer(canvas,
                                 style_img,
                                 resolution=pixel_resolution,
                                 num_steps=num_steps_pixel,
                                 content_weight=1.0,
                                 style_weight=10000.0)

    print('Pixel optimization:')
    canvas = pixel_optim.optimize()
    return canvas


class BrushstrokeOptimizer:

    def __init__(self,
                 content_img,                              # Content image (PIL.Image).
                 style_img,                                # Style image (PIL.Image).
                 draw_curve_position_path = None,          # Set of points that represent the drawn curves, denoted as P_i in Sec. B of the paper (str).
                 draw_curve_vector_path   = None,          # Set of tangent vectors for the points of the drawn curves, denoted as v_i in Sec. B of the paper (str).
                 draw_strength            = 100,           # Strength of the influence of the drawn curves, denoted L in Sec. B of the paper (int).
                 resolution               = 512,           # Resolution of the canvas (int).
                 num_strokes              = 5000,          # Number of brushstrokes (int).
                 num_steps                = 100,           # Number of optimization steps (int).
                 S                        = 10,            # Number of points to sample on each curve, see Sec. 4.2.1 of the paper (int).
                 K                        = 20,            # Number of brushstrokes to consider for each pixel, see Sec. C.2 of the paper (int).
                 canvas_color             = 'gray',        # Color of the canvas (str).
                 width_scale              = 0.1,           # Scale parameter for the brushstroke width (float).
                 length_scale             = 1.1,           # Scale parameter for the brushstroke length (float).
                 content_weight           = 1.0,           # Weight for the content loss (float).
                 style_weight             = 3.0,           # Weight for the style loss (float).
                 tv_weight                = 0.008,         # Weight for the total variation loss (float).
                 draw_weight              = 100.0,         # Weight for the drawing projection loss (float)
                 curviture_weight         = 4.0,           # Weight for the curviture loss (float).
                 streamlit_pbar           = None,          # Progressbar for streamlit app (obj).
                 dtype                    = 'float32',      # Data type (str).
                 init = "sp",
                 init_prob = None,
                 offset=0.5,
                 init_width=None,
                 width_fixed = False,
                 optim_rate=0.1
                ):
    
        self.draw_strength = draw_strength
        self.draw_weight = draw_weight
        self.resolution = resolution
        self.num_strokes = num_strokes
        self.num_steps = num_steps
        self.S = S
        self.K = K
        self.canvas_color = canvas_color
        self.width_scale = width_scale
        self.length_scale = length_scale
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.curviture_weight = curviture_weight
        self.streamlit_pbar = streamlit_pbar
        self.dtype = dtype
        
        self.init = init
        self.init_prob = init_prob
        self.offset = offset
        self.init_width = init_width
        self.width_fixed = width_fixed
        self.optim_rate = optim_rate

        # Set canvas size (set smaller side of content image to 'resolution' and scale other side accordingly)
        W, H = content_img.size
        if H < W:                                                                                        
            new_H = resolution                                                                           
            new_W = int((W / H) * new_H)
            next_W = int(2*(W / H) * new_H)
            next_H = 2*resolution
        else:                                                                                            
            new_W = resolution
            new_H = int((H / W) * new_W)                                                                
            next_H = int(2*(H / W) * new_W) 
            next_W = 2*resolution
                                                                                                         
        self.canvas_height = new_H
        self.canvas_width = new_W

        content_img = content_img.resize((self.canvas_width, self.canvas_height))
        style_img = style_img.resize((self.canvas_width, self.canvas_height))
        if self.init_prob is not None:
            self.init_prob = skimage.transform.resize(self.init_prob,(new_H,new_W,),order=3)
        if isinstance(self.canvas_color,str) is False:
            self.canvas_color = skimage.transform.resize(self.canvas_color,(new_H,new_W,),order=3)
        content_img = np.array(content_img).astype(self.dtype)
        style_img = np.array(style_img).astype(self.dtype)

        content_img /= 255.0
        style_img /= 255.0
        
        self.content_img_np = content_img
        self.style_img_np = style_img

        if draw_curve_position_path is not None and draw_curve_vector_path is not None:
            self.draw_curve_position_np = np.load(draw_curve_position_path)
            self.draw_curve_vector_np = np.load(draw_curve_vector_path)
            self.draw_curve_position_np[..., 0] *= self.canvas_width
            self.draw_curve_position_np[..., 1] *= self.canvas_height

        ckpt_path = utils.download_weights(url='https://www.dropbox.com/s/hv7b4eajrj7isyq/vgg_weights.pickle?dl=1',
                                           name='vgg_weights.pickle')
        self.vgg = networks.VGG(ckpt_path=ckpt_path)

    def optimize(self):
        self._initialize()
        self._render()
        self._losses()
        self._optimizer()


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            steps = trange(self.num_steps, desc='', leave=True)
            for step in steps:
                
                I_, loss_dict_, params_dict_, _,s,e,c,l,colours,lm,w = \
                    sess.run(fetches=[self.I, 
                                      self.loss_dict, 
                                      self.params_dict, 
                                      self.optim_step_with_constraints,
                                      self.curve_s,
                                      self.curve_e,
                                      self.curve_c,
                                      self.location,
                                      self.color,
                                      self.lossmaps[-1],
                                      self.width],
                             options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True)
                            )

                steps.set_description(f'content_loss: {loss_dict_["content"]:.6f}')
                #s = ''
                #for key in loss_dict_:
                #    loss = loss_dict_[key]
                #    s += key + f': {loss_dict_[key]:.4f}, '
                #steps.set_description(s[:-2])
                #print(s)

                steps.refresh()
                if self.streamlit_pbar is not None: self.streamlit_pbar.update(1)
        return Image.fromarray(np.array(np.clip(I_, 0, 1) * 255, dtype=np.uint8)),s,e,c,l,colours,lm,w

    def _initialize(self):
        location, s, e, c, width, color = utils.initialize_brushstrokes(self.content_img_np, 
                                                                        self.num_strokes, 
                                                                        self.canvas_height, 
                                                                        self.canvas_width, 
                                                                        self.length_scale, 
                                                                        self.width_scale,
                                                                        init=self.init,
                                                                        init_prob = self.init_prob,
                                                                        offset=self.offset,
                                                                        init_width=self.init_width)

        self.curve_s = tf.Variable(name='curve_s', initial_value=s, dtype=self.dtype)
        self.curve_e = tf.Variable(name='curve_e', initial_value=e, dtype=self.dtype)
        self.curve_c = tf.Variable(name='curve_c', initial_value=c, dtype=self.dtype)
        self.color = tf.Variable(name='color', initial_value=color, dtype=self.dtype)
        self.location = tf.Variable(name='location', initial_value=location, dtype=self.dtype)
        self.width = tf.Variable(name='width', initial_value=width, dtype=self.dtype)
        self.content_img = tf.constant(name='content_img', value=self.content_img_np, dtype=self.dtype)
        self.style_img = tf.constant(name='style_img', value=self.style_img_np, dtype=self.dtype)

        if hasattr(self, 'draw_curve_position_np') and hasattr(self, 'draw_curve_vector_np'):
            self.draw_curve_position = tf.constant(name='draw_curve_position', value=self.draw_curve_position_np, dtype=self.dtype)
            self.draw_curve_vector = tf.constant(name='draw_curve_vector', value=self.draw_curve_vector_np, dtype=self.dtype)

        self.params_dict = {'location': self.location, 
                            'curve_s': self.curve_s, 
                            'curve_e': self.curve_e, 
                            'curve_c': self.curve_c, 
                            'width': self.width, 
                            'color': self.color}

    def _render(self):
        curve_points,locs,colors,widths = ops.sample_quadratic_bezier_curve2(s=self.curve_s + self.location,
                                                         e=self.curve_e + self.location,
                                                         c=self.curve_c + self.location,
                                                         colors = self.color,
                                                         widths = self.width,
                                                         num_points=self.S,
                                                         dtype=self.dtype)

        self.I = ops.renderer(curve_points, 
                              locs, 
                              colors, 
                              widths, 
                              self.canvas_height, 
                              self.canvas_width, 
                              self.K, 
                              canvas_color=self.canvas_color, 
                              dtype=self.dtype)

    def _losses(self):
        # resize images to save memory
        rendered_canvas_resized = \
            tf.image.resize_nearest_neighbor(images=ops.preprocess_img(self.I),
                                             size=(int(self.canvas_height), int(self.canvas_width)))
        rendered_canvas_resized2 = \
            tf.image.resize_nearest_neighbor(images=ops.preprocess_img(self.I),
                                             size=(int(2*self.canvas_height), int(2*self.canvas_width)))
        content_img_resized = \
            tf.image.resize_nearest_neighbor(images=ops.preprocess_img(self.content_img),
                                             size=(int(self.canvas_height), int(self.canvas_width)))
        
        content_img_resized2 = \
            tf.image.resize_nearest_neighbor(images=ops.preprocess_img(self.content_img),
                                             size=(int(2*self.canvas_height), int(2*self.canvas_width)))

        style_img_resized = \
            tf.image.resize_nearest_neighbor(images=ops.preprocess_img(self.style_img),
                                             size=(int(self.canvas_height), int(self.canvas_width)))

        self.loss_dict = {}
        canvas_feats = self.vgg.extract_features(rendered_canvas_resized)
        content_feats = self.vgg.extract_features(content_img_resized)
        layers=['conv4_2', 'conv5_2']
        self.loss_dict['content'] = ops.content_loss(canvas_feats,
                                                     content_feats,
                                                     #layers=['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2'],
                                                     layers=layers,
                                                     weights=[1, 1],
                                                     scale_by_y=True)
        canvas_feats2 = self.vgg.extract_features(rendered_canvas_resized2)
        content_feats2 = self.vgg.extract_features(content_img_resized2)
        self.lossmaps = []
        for layer in layers:
            self.lossmaps.append(tf.reduce_mean(tf.square(canvas_feats2[layer]-content_feats2[layer]) * tf.minimum(content_feats2[layer], tf.sigmoid(content_feats2[layer])),-1))
        self.loss_dict['content'] *= self.content_weight

        #self.loss_dict['style'] = ops.style_loss(self.vgg.extract_features(rendered_canvas_resized),
        #                                         self.vgg.extract_features(style_img_resized),
        #                                         layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
        #                                         weights=[1, 1, 1, 1, 1])
        #self.loss_dict['style'] *= self.style_weight

        self.loss_dict['curviture'] = ops.curviture_loss(self.curve_s, self.curve_e, self.curve_c)
        self.loss_dict['curviture'] *= self.curviture_weight

        self.loss_dict['tv'] = ops.total_variation_loss(x_loc=self.location, s=self.curve_s, e=self.curve_e, K=10)
        self.loss_dict['tv'] *= self.tv_weight

        if hasattr(self, 'draw_curve_position') and hasattr(self, 'draw_curve_vector'):
            self.loss_dict['drawing'] = ops.draw_projection_loss(self.location, 
                                                                 self.curve_s, 
                                                                 self.curve_e, 
                                                                 self.draw_curve_position, 
                                                                 self.draw_curve_vector, 
                                                                 self.draw_strength)
            self.loss_dict['drawing'] *= self.draw_weight


    def _optimizer(self):
        loss = tf.constant(0.0)
        for key in self.loss_dict:
            loss += self.loss_dict[key]
        
        step_ops = []
        var_list = [self.location, self.curve_s, self.curve_e, self.curve_c, self.width]
        if self.width_fixed:
            var_list = [self.location, self.curve_s, self.curve_e, self.curve_c]
        optim_step = tf.train.AdamOptimizer(self.optim_rate).minimize(
            loss=loss, 
            var_list=var_list)
        step_ops.append(optim_step)
        #optim_step_color = tf.train.AdamOptimizer(0.01).minimize(
        #    loss=self.loss_dict['style'],
        #    var_list=self.color)
        #step_ops.append(optim_step_color)

        # constraint parameters to certain range
        with tf.control_dependencies(step_ops.copy()):
            step_ops.append(tf.assign(self.color, tf.clip_by_value(self.color, 0, 1)))
            coord_x, coord_y = tf.gather(self.location, axis=-1, indices=[0]), tf.gather(self.location, axis=-1, indices=[1])
            coord_clip = tf.concat([tf.clip_by_value(coord_x, 0, self.canvas_height), tf.clip_by_value(coord_y, 0, self.canvas_width)], axis=-1)
            step_ops.append(tf.assign(self.location, coord_clip))
            if self.width_fixed == False:
              step_ops.append(tf.assign(self.width, tf.nn.relu(self.width)))
            else:
              step_ops.append(tf.assign(self.width, tf.clip_by_value(self.width,self.init_width-0.1,self.init_width+0.1)))
        self.optim_step_with_constraints = tf.group(*step_ops)


class PixelOptimizer:

    def __init__(self,
                 canvas,                              # Canvas (PIL.Image).
                 style_img,                           # Style image (PIL.Image).
                 resolution          = 1024,          # Resolution of the canvas.
                 num_steps           = 2000,          # Number of optimization steps.
                 content_weight      = 1.0,           # Weight for the content loss.
                 style_weight        = 10000.0,       # Weight for the style loss.
                 tv_weight           = 0.0,           # Weight for the total variation loss.
                 streamlit_pbar      = None,          # Progressbar for streamlit app (obj).
                 dtype               = 'float32'      # Data type.
                ):
    
        self.resolution = resolution
        self.num_steps = num_steps
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.streamlit_pbar = streamlit_pbar
        self.dtype = dtype

        # Set canvas size (set smaller side of content image to 'resolution' and scale other side accordingly)
        W, H = canvas.size
        if H < W:                                                                                        
            new_H = resolution                                                                           
            new_W = int((W / H) * new_H)
        else:                                                                                            
            new_W = resolution
            new_H = int((H / W) * new_W)                                                                 
                                                                                                         
        self.canvas_height = new_H              
        self.canvas_width = new_W

        canvas = canvas.resize((self.canvas_width, self.canvas_height))
        style_img = style_img.resize((self.canvas_width, self.canvas_height))
        
        canvas = np.array(canvas).astype(self.dtype)
        style_img = np.array(style_img).astype(self.dtype)

        canvas /= 255.0
        style_img /= 255.0

        self.canvas_np = canvas
        self.content_img_np = canvas
        self.style_img_np = style_img

        ckpt_path = utils.download_weights(url='https://www.dropbox.com/s/hv7b4eajrj7isyq/vgg_weights.pickle?dl=1',
                                           name='vgg_weights.pickle')
        self.vgg = networks.VGG(ckpt_path=ckpt_path)

    def optimize(self):
        self._initialize()
        self._losses()
        self._optimizer()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            steps = trange(self.num_steps, desc='', leave=True)
            for step in steps:
                canvas_, loss_dict_, _ = \
                    sess.run(fetches=[self.canvas, 
                                      self.loss_dict, 
                                      self.optim_step_with_constraints],
                             options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True)
                            )

                s = ''
                for key in loss_dict_:
                    loss = loss_dict_[key]
                    s += key + f': {loss_dict_[key]:.6f}, '

                steps.set_description(s[:-2]) 
                steps.refresh()
                if self.streamlit_pbar is not None: self.streamlit_pbar.update(1)
        return Image.fromarray(np.array(np.clip(canvas_, 0, 1) * 255, dtype=np.uint8))

    def _initialize(self):
        self.canvas = tf.Variable(name='canvas', initial_value=self.canvas_np, dtype=self.dtype)
        self.content_img = tf.constant(name='content_img', value=self.content_img_np, dtype=self.dtype)
        self.style_img = tf.constant(name='style_img', value=self.style_img_np, dtype=self.dtype)

    def _losses(self):
        # resize images to save memory
        rendered_canvas_resized = \
            tf.image.resize_nearest_neighbor(images=ops.preprocess_img(self.canvas),
                                             size=(int(self.canvas_height), int(self.canvas_width)))

        content_img_resized = \
            tf.image.resize_nearest_neighbor(images=ops.preprocess_img(self.content_img),
                                             size=(int(self.canvas_height), int(self.canvas_width)))

        style_img_resized = \
            tf.image.resize_nearest_neighbor(images=ops.preprocess_img(self.style_img),
                                             size=(int(self.canvas_height), int(self.canvas_width)))

        self.loss_dict = {}
        self.loss_dict['content'] = ops.content_loss(self.vgg.extract_features(rendered_canvas_resized),
                                                     self.vgg.extract_features(content_img_resized),
                                                     layers=['conv1_2_pool', 'conv2_2_pool', 'conv3_3_pool', 'conv4_3_pool', 'conv5_3_pool'],
                                                     weights=[1, 1, 1, 1, 1])
        self.loss_dict['content'] *= self.content_weight

        self.loss_dict['style'] = ops.style_loss(self.vgg.extract_features(rendered_canvas_resized),
                                                 self.vgg.extract_features(style_img_resized),
                                                 layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                                                 weights=[1, 1, 1, 1, 1])
        self.loss_dict['style'] *= self.style_weight

        self.loss_dict['tv'] = ((tf.nn.l2_loss(self.canvas[1:, :, :] - self.canvas[:-1, :, :]) / self.canvas.shape.as_list()[0]) +
                                (tf.nn.l2_loss(self.canvas[:, 1:, :] - self.canvas[:, :-1, :]) / self.canvas.shape.as_list()[1]))
        self.loss_dict['tv'] *= self.tv_weight

    def _optimizer(self):
        loss = tf.constant(0.0)
        for key in self.loss_dict:
            loss += self.loss_dict[key]
        
        step_ops = []
        optim_step = tf.train.AdamOptimizer(0.01).minimize(loss=loss, var_list=self.canvas)
        step_ops.append(optim_step)

        # constraint parameters to certain range
        with tf.control_dependencies(step_ops.copy()):
            step_ops.append(tf.assign(self.canvas, tf.clip_by_value(self.canvas, 0, 1)))

        self.optim_step_with_constraints = tf.group(*step_ops)



