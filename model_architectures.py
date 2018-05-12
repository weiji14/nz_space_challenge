import affine
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import math
import numpy as np
import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose, SeparableConv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, AlphaDropout, Concatenate
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import add, concatenate
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.utils import multi_gpu_model, plot_model
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.patches
import rasterio
import scipy.spatial
import scipy.stats
import gym
from gym.utils import seeding

#######################################################
##               Crevasse Finder model               ##
#######################################################
def crevasse_finder_model(img_height=256, img_width=256, img_channels=1, tensorboard_images=False):
    '''
    Modified from https://keunwoochoi.wordpress.com/2017/10/11/u-net-on-keras-2-0/
    Architecture inspired by https://blog.deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/
    '''
    encoder_exps = [5,6,7,8,9]   #the n-th deep channel's exponent i.e. 2**n 16,32,64,128,256
    decoder_exps = [8,7,6,5]   #reversed order of exponents e.g. 128,64,32,16
    k_size = (3, 3)                     #size of filter kernel
    k_init = 'lecun_normal'             #kernel initializer
    activation = 'selu'
    
    ## Standard layers we use a lot in the model
    dropout = lambda rate: AlphaDropout(rate=rate)   #lambda rate: Dropout(rate=rate)
    conv2d = lambda filters: SeparableConv2D(filters=filters, kernel_size=k_size,
                                              activation=activation, padding='same',
                                              depthwise_initializer=k_init, pointwise_initializer=k_init,
                                              use_bias=False)
    
    if K.image_data_format() == 'channels_first':
        ch_axis = 1
        input_shape = (img_channels, img_height, img_width)
    elif K.image_data_format() == 'channels_last':
        ch_axis = 3
        input_shape = (img_height, img_width, img_channels)

    inp = Input(shape=input_shape)
    if tensorboard_images == True:
        tf.summary.image(name='input', tensor=inp)
    
    # encoder
    enc_ = []
    enc = inp
    #print(encoder_exps)
    for i, n_ch in enumerate(encoder_exps, start=1):  #loop from 1, 2, 3, ..., len(n_ch_exps)
        with K.name_scope(f'Encode_block_{i}'):
            enc = BatchNormalization()(enc)
            enc = conv2d(filters=2**n_ch)(enc)
            enc = dropout(rate=0.2)(enc)
            enc = conv2d(filters=2**n_ch)(enc)
            enc_.append(enc)
            if i < len(encoder_exps):  #do not run max pooling on the last encoding/downsampling step
                enc = MaxPooling2D(pool_size=(2,2))(enc)  #strides = pool_size if strides is not set
                #enc[i] = Conv2D(filters=2**n_ch, kernel_size=k_size, strides=(2,2), activation=activation, padding='same', kernel_initializer=k_init)(enc[i])
    
    # decoder
    dec = enc
    #print(decoder_exps)
    for j, n_ch in enumerate(decoder_exps, start=1):
        with K.name_scope(f'Decode_block_{j}'):
            dec = Conv2DTranspose(filters=2**n_ch, kernel_size=k_size, strides=(2,2), activation=activation, padding='same', kernel_initializer=k_init)(dec)
            ## Bilinear upsampling + a Conv2D
            #new_width = int(img_width/(2**(len(decoder_exps)-j)))
            #new_height = int(img_height/(2**(len(decoder_exps)-j)))
            #print(new_width, new_height)
            #resize_function = lambda images: K.tf.image.resize_bilinear(images=images, size=[new_height,new_width], name=f'tf_bilinear_upsampling_{j}')
            #dec_.append(dec)
            #dec = Lambda(function=resize_function, name=f'bilinear_upsampling_{j}')(dec_[-1])
            #dec = conv2d(filters=2**n_ch)(dec)
            
            dec = Concatenate(axis=ch_axis)([dec, enc_[-(j+1)]])  #Use deepcopy?, see https://github.com/keras-team/keras/issues/5972
            dec = BatchNormalization()(dec)
            dec = conv2d(filters=2**n_ch)(dec)
            dec = dropout(rate=0.2)(dec)
            dec = conv2d(filters=2**n_ch)(dec)
            
    outp = SeparableConv2D(filters=1, kernel_size=k_size, activation='sigmoid', padding='same',
                           depthwise_initializer=k_init, pointwise_initializer=k_init, use_bias=False)(dec)
    if tensorboard_images == True:
        tf.summary.image(name='output', tensor=outp)
    
    model = Model(inputs=[inp], outputs=[outp])
    
    return model



#######################################################
##                Route Finder models                ##
#######################################################
def route_finder_model(observation_input_shape:tuple=(256,256,1)):
    k_init = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
    # Shared Convolutional and Dense Layer(s)
    with K.tf.variable_scope('actor_critic_shared_conv_layers'):
        inp = Input(shape=observation_input_shape)
        assert(inp.shape.ndims == 4)  #needs to be shape like (?,4,4,1) for 4x4 grid, (?,8,8,1) for 8x8 grid
        X = Conv2D(filters=16, kernel_size=(8,8), strides=(4,4), kernel_initializer=k_init, padding='same', activation='relu')(inp)
        X = Conv2D(filters=32, kernel_size=(4,4), strides=(2,2), kernel_initializer=k_init, padding='same', activation='relu')(X)
        X = Flatten()(X)
        X = Dense(units=256, kernel_initializer=k_init, activation='relu')(X)
    
    #Actor's Policy output (probability of taking each possible action)
    with K.tf.variable_scope('actor_output'):
        actor_output = Dense(units=4, kernel_initializer=k_init, activation='softmax', name='actor_output')(X)
    
    actor_model = Model(inputs=inp, outputs=actor_output)
    
    return actor_model



#######################################################
##          Crevasse Crosser Gym Environment         ##
#######################################################
#https://stackoverflow.com/questions/45068568/is-it-possible-to-create-a-new-gym-environment-in-openai/47132897#47132897
class CrevasseCrosserEnv(gym.Env):
    """
    An OpenAI gym compatible custom environment inspired from the FrozenLake env!
    
    The map should be a 2D numpy array, with values on each xy position indicating the
    penalty for stepping on to that position. The centre pixels, specifically a 2x2 square,
    will be where our player is assumed to be located. Moving around the map environment 
    is done using a scrolling method, similar to classic RPG games! Not sure if I explained
    this intuitively enough :P
    
    Anyways, E.g. if values range from 0 to 1
    Values close to '1' would indicate a hole or 'H' in the FrozenLake env
    Values close to '0' would indicate the goal or 'G' in the FrozenLake env
    Values in between '0.25-0.75' would be the free spot or 'F' in the FrozenLake env, note that we penalize steps!
    There is not really a starting point 'S', we always assume the player is in the centre
    """
    
    def __init__(self, map_env:np.ndarray=None, observation_window:tuple=(256,256,1), pad_border:bool=True,
                 transform:affine.Affine=None, start_xy:tuple=None, end_xy:tuple=None):
        
        self.map_env = map_env
        assert(isinstance(self.map_env, np.ndarray))  #check that we are passing in a numpy array
        assert(map_env.ndim==3)  #check that we have a map_env of shape (height, width, channels)
                
        self.observation_window = observation_window
        assert(len(self.observation_window) == 3)  #check that we have a observation window of shape (height, width, channels)
        assert(self.observation_window[0]%2 == 0)  #check that our observation window height is an even number
        assert(self.observation_window[1]%2 == 0)  #check that our observation window width is an even number
        
        self.map_env_shape = self.map_env.shape    #create an alias to the map_env shape
        assert(self.map_env_shape[-1] == self.observation_window[-1])  #check that the no. of channels in our map_env matches the no. of channels in our observation_window
        assert(self.map_env_shape[-1] == 1)  #check that we only have a one band/channel input, TODO remove when moving to more bands!
        
        self.info = None      #dictionary to store geographic coordinates of our player
        self.gameover = None  #setting to store whether game round is over
        self.total_steps = 0  #store a counter of steps taken by the player
        self.centrepoint = {'pixel_midy':None, 'pixel_midx': None}  #initialize this observation centrepoint variable

        ## number of Actions (discrete)
        ## Define what our agent can do, i.e. move left, down, right, up
        self.nA = 4
        self.action_space = gym.spaces.Discrete(self.nA)
        
        self.transform = transform
        self.start_xy = start_xy
        self.end_xy = end_xy
        
        self.random_start_xy = True if self.start_xy is None else False
        self.random_end_xy = True if self.end_xy is None else False
        
        try:
            assert(self.transform is not None)     #check that affine transform is available, so coordinates can be converted!!
        except AssertionError:
            self.transform = rasterio.transform.Affine(100, 0, -0,
                                                        0, -100, -0)  #100m pixel x and y size, 0 rotation and (0,0) origin
            print('WARN: Geo Affine transformation not set for Gym Env, using default instead: {self.transform}')
        
        self.pad_border = pad_border               #whether or not to pad border with direction hints
        self.current_observation = self.reset()    #reset our current observation to a random or designated initial state
            
        if self.start_xy is not None:
            assert(len(self.start_xy) == 2)    #check that we are passing in 2 numbers (x_coordinate, y_coordinate)
        if self.end_xy is not None:
            assert(len(self.end_xy) == 2)      #check that we are passing in 2 numbers (x_coordinate, y_coordinate)
            self.geodist_to_goal = scipy.spatial.distance.euclidean(u=(self.centrepoint['geo_midx'], self.centrepoint['geo_midy']),
                                                                    v=self.end_xy)
        
    def reset(self, goal_difficulty:int=8):
        """
        Function to reset the entire environment
        
        if self.start_xy is None:
            Set our player at some random place on the map
        elif self.start_xy is (x_coord, y_coord):
            Set our player at the designated place
        """
        self.info = None
        self.gameover = False
        self.lastaction = None
        self.total_steps = 0
        
        ## Get x and y starting positions (image coordinates), either by random or from input start_xy
        if self.random_start_xy == True:  # Get random x and y starting positions (img coords) from the map_env that are not too close to the border
            pixel_midy = np.random.randint(low=0+4*self.observation_window[0],
                                           high=self.map_env.shape[0]-4*self.observation_window[0])
            pixel_midx = np.random.randint(low=0+4*self.observation_window[1],
                                           high=self.map_env.shape[1]-4*self.observation_window[1])
            self.start_xy = rasterio.transform.xy(transform=self.transform,
                                                       cols=pixel_midx,
                                                       rows=pixel_midy,
                                                       offset='lr')  #convert image coordinates to geographic coordinates
        else:
            pixel_midy, pixel_midx = rasterio.transform.rowcol(transform=self.transform,
                                                               xs=self.start_xy[0], ys=self.start_xy[1],
                                                               op=math.ceil)  #math.ceil rounds to bottom right in image coords
            assert(0 <= pixel_midy < self.map_env.shape[0])  #check (roughly) that our y is within the map environment
            assert(0 <= pixel_midx < self.map_env.shape[1])  #check (roughly) that our x is within the map environment
                
        if self.random_end_xy == True:
            '''We will set the endpoint as some random position +/-goal_difficulty (e.g. 8) pixels or so from the starting pixel'''
            end_pixel_midy = np.random.uniform(low=pixel_midy-goal_difficulty, high=pixel_midy+goal_difficulty)
            end_pixel_midx = np.random.uniform(low=pixel_midx-goal_difficulty, high=pixel_midx+goal_difficulty)
            self.end_xy = rasterio.transform.xy(transform=self.transform,
                                                       cols=end_pixel_midx,
                                                       rows=end_pixel_midy,
                                                       offset='lr')  #convert image coordinates to geographic coordinates
        
        self.current_observation, self.centrepoint = self.set_observation(pixel_midy=pixel_midy, pixel_midx=pixel_midx)
        
        return self.current_observation  #return initial observation
    
    
    def set_observation(self, pixel_midy:int, pixel_midx:int):
        """
        Crop a 2D slice out of our map environment given an input midpoint
        Note that the midpoint should reference the bottom right corner
        of the pixel, so that the array slicing method will be correct!!
        """
        pixel_y0 = pixel_midy - int(self.observation_window[0]/2)
        pixel_y1 = pixel_midy + int(self.observation_window[0]/2)
        pixel_x0 = pixel_midx - int(self.observation_window[1]/2)
        pixel_x1 = pixel_midx + int(self.observation_window[1]/2)
        
        observation = self.map_env[pixel_y0:pixel_y1,
                                   pixel_x0:pixel_x1, :]
        try:
            assert(observation.shape == self.observation_window)  #check that obsevation is within map environment!
        except:
            print(f"{observation.shape} is less than {self.observation_window}")
            raise
        
        centrepoint = {'pixel_midy': pixel_midy, 'pixel_midx': pixel_midx}
        
        if self.transform is not None:
            geo_midx, geo_midy = rasterio.transform.xy(transform=self.transform,
                                                       cols=pixel_midx,
                                                       rows=pixel_midy,
                                                       offset='lr')  #convert image coordinates to geographic coordinates
            centrepoint.update({'geo_midy': geo_midy, 'geo_midx': geo_midx})
        else:
            centrepoint.update({'geo_midy': None, 'geo_midx': None})
        
        
        if self.pad_border == True:
            '''
            Pad the borders around our observation window to let the agent
            know via a hot/cold method where our destination lies
              
               ---- ---- 
               ---- ---- 
             ||         ||
             ||         ||
                   A
             ||         ||
             ||         ||
               ---- ---- 
               ---- ---- 
            '''
            
            #print(f'current_xy: {geo_midx,geo_midy}, end_xy: {self.end_xy}' )
            x_diff = self.end_xy[0] - geo_midx
            y_diff = self.end_xy[1] - geo_midy
            #print(f'x_diff {x_diff}, y_diff {y_diff}')
            
            padded_observation = np.full_like(a=observation, dtype=observation.dtype, fill_value=1)
            #half_val = np.mean(a=[np.finfo(observation.dtype).max, np.info(observation.dtype).min])  #0.5 for float32 range(0,1)
            
            if x_diff <= 0: #i.e. if our goal lies to the left of the image
                padded_observation[0:self.observation_window[0], :self.observation_window[0]//2, :] -= np.array([0.5], dtype=observation.dtype)  #decrease penalty of left half 
            if x_diff >= 0: #i.e. if our goal lies to the right of the image
                padded_observation[0:self.observation_window[0], self.observation_window[0]//2:, :] -= np.array([0.5], dtype=observation.dtype)  #decrease penalty of right half
            
            if y_diff <= 0: #i.e. if our goal lies to the bottom of the image
                padded_observation[self.observation_window[0]//2:, 0:self.observation_window[1], :] -= np.array([0.5], dtype=observation.dtype)  #decrease penalty of bottom half
            if y_diff >= 0:  #i.e. if our goal lies to the top of the image
                padded_observation[:self.observation_window[0]//2, 0:self.observation_window[1], :] -= np.array([0.5], dtype=observation.dtype)  #decrease penalty of top half
        
            #plt.imshow(-padded_observation[:,:,0], cmap='BrBG')
            
            padded_observation[4:-4, 4:-4, :] = observation[4:-4,4:-4,:]   #fill in centre of padded_observation with actual observation
            observation = padded_observation
            
        return observation, centrepoint
    
    
    def step(self, action):
        """
        Moves our player around the environment by applying an action
        
        Returns:
        (observation, reward, done, info)
        """
        if self.gameover == True:
            return (self.current_observation, 0, True, {})
        
        ## Get new OBSERVATION after making an action
        new_pixel_midy = self.centrepoint['pixel_midy']
        new_pixel_midx = self.centrepoint['pixel_midx']
        
        #try:
        #    assert(isinstance(action, int))  #used for standard dqn where action is an integer
        #except AssertionError:
        #    action = np.argmax(a=action)  #used for actor-critic environment where action is a list of probs
        
        if action == 0: #Left <-
            if new_pixel_midx - self.observation_window[1]/2 - 2 > 0:
                new_pixel_midx -= 2
                self.total_steps += 1
        elif action == 1: #Down v
            if new_pixel_midy + self.observation_window[0]/2 + 2 < self.map_env_shape[0]:
                new_pixel_midy += 2
                self.total_steps += 1
        elif action == 2: #Right ->
            if new_pixel_midx + self.observation_window[1]/2 + 2 < self.map_env_shape[1]:
                new_pixel_midx += 2
                self.total_steps += 1
        elif action == 3: # Up ^
            if new_pixel_midy - self.observation_window[0]/2 - 2 > 0:
                new_pixel_midy -= 2
                self.total_steps += 1
        
        self.current_observation, self.centrepoint = self.set_observation(pixel_midy=new_pixel_midy,
                                                                          pixel_midx=new_pixel_midx)
        
        ## Get REWARD from centre 2x2 grid square
        '''     0 , 10 , 20 , 30 , 40 , 50 , 60 , 70 , 80 , 90 , 100 = percentiles
                                           |
                        <- good safe spots | bad crevasse holes ->
        reward:                  -0.001    |    -1                             '''
        
        midsquare_vals = self.map_env[self.centrepoint['pixel_midy']-1:self.centrepoint['pixel_midy']+1,
                                      self.centrepoint['pixel_midx']-1:self.centrepoint['pixel_midx']+1, :]
        midsquare_max = np.max(midsquare_vals)  #get maximum of the midsquare pixels
        percentileofscore = scipy.stats.percentileofscore(a=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                                          score=midsquare_max)
        midsquare_rank = 10 * int(math.floor(percentileofscore/10.0))  #rank from 0 (good goal) to 100 (bad hole)
        assert(midsquare_rank % 10 == 0)
        
        S, H = (-0.001, -1)  #base rewards corresponding to Safe, Hole
        reward = [S,S,S,S,S,S,H,H,H,H,H][midsquare_rank//10]  #use python indexing to get base reward
        #print(midsquare_rank, reward)
        
        ## Get DONE (or not) status 
        if midsquare_rank >= 60:  #i.e. if we're in a crevasse!
            done = self.gameover = True
        else:                     #if we haven't fell in to a crevasse (yet)
            assert(self.end_xy is not None)  #Make sure that an endpoint is set!!
            geo_resolution = self.transform.a  #equal to self.transform[0], assuming x and y resolution are the same
            current_geox, current_geoy = (self.centrepoint['geo_midx'], self.centrepoint['geo_midy'])
                
            new_geodist_to_goal = scipy.spatial.distance.euclidean(u=(current_geox, current_geoy),
                                                                       v=self.end_xy)
            if new_geodist_to_goal <= 2*geo_resolution:   #when we are within 2x the spatial resolution of the endpoint
                bonus = min(+1 + abs(S*self.total_steps), 2)      #arbitrarily large bonus for reaching the goal
                reward += bonus
                done = self.gameover = True
            elif new_geodist_to_goal > 2*geo_resolution:  #when we are still somewhat far from our goal
                done = False
                
            self.geodist_to_goal = new_geodist_to_goal
           
    
        ## Get INFO about coordinates of midpoint
        self.info = {'total_steps': self.total_steps,
                     'geo_x': self.centrepoint['geo_midx'],  #may be None if self.transform is None
                     'geo_y': self.centrepoint['geo_midy'],  #may be None if self.transform is None
                     'pixel_x': self.centrepoint['pixel_midx'], 
                     'pixel_y': self.centrepoint['pixel_midy']
                     }
        if self.end_xy is not None:
            self.info.update({'geodist_to_goal': self.geodist_to_goal})

        self.lastaction = action
        return (self.current_observation, reward, done, self.info)
    
    
    def render(self, mode='human'):
        """
        Matplotlib function to plot the current observation window
        with the player position in the middle as a 2x2 grid box
        """
        fig, axarr = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(10,5))
        
        # Left plot entire observation view
        axarr[0, 0].imshow(X=-self.current_observation[:,:,0], cmap='BrBG')
        upper_left_y = self.observation_window[0]/2 - 1 - 0.5 
        upper_left_x = self.observation_window[1]/2 - 1 - 0.5
        axarr[0, 0].add_patch(matplotlib.patches.Rectangle(xy=(upper_left_x,upper_left_y), width=2, height=2,
                                                           fill=False, edgecolor='red', linewidth=2))
        
        # Right plot 4x zoom in view of observation's centre
        crop_y0 = self.observation_window[0]//2 - self.observation_window[0]//16
        crop_y1 = self.observation_window[0]//2 + self.observation_window[0]//16
        crop_x0 = self.observation_window[1]//2 - self.observation_window[1]//16
        crop_x1 = self.observation_window[1]//2 + self.observation_window[1]//16
        axarr[0, 1].imshow(X=-self.current_observation[crop_y0:crop_y1,crop_x0:crop_x1,0], cmap='BrBG')
        upper_left_y = (crop_y1-crop_y0)/2 - 1 - 0.5 
        upper_left_x = (crop_x1-crop_x0)/2 - 1 - 0.5
        axarr[0, 1].add_patch(matplotlib.patches.Rectangle(xy=(upper_left_x,upper_left_y), width=2, height=2,
                                                           fill=False, edgecolor='red', linewidth=2))
        
        
        if self.lastaction is not None:
            actionTaken = ["Left","Down","Right","Up"][self.lastaction]
            print(f"  ({actionTaken}, {self.info})")
        else:
            print("")
        return plt.show()
    
    
    def seed(self, seed=None):
        """
        Just a simple function to set the seed of our gym environment
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed=seed)
        return [seed]