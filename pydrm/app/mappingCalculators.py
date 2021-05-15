import numpy as np
import tensorflow as tf
from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

def QR_window_generator(rx, ry, wsx, wsy):
    xinds = np.floor(rx//wsx)
    resx = rx - wsx*xinds
    yinds = np.floor(ry//wsy)
    resy = ry - wsy*yinds
    for kx in range(int(xinds)):
        xs = kx*wsx
        xe = xs + wsx if xs + wsx <= rx else xs + resx
        for ky in range(int(yinds)):
            ys = ky*wsy
            ye = ys + wsy if ys + wsy <= ry else ys + resy
            yield xs, xe, ys, ye

class BiTexDiscriminator():
    def __init__(self, master, master_pbar, master_object, host_canvas, host_master_canvas):
        self.master = master
        self.pbar = master_pbar
        self.master_object = master_object
        self.host_canvas = host_canvas
        self.canvas = host_master_canvas
        self.total_epochs = 0
        self.train_losses = []
        self.valid_losses = []
        
        # self.compressor_to_fit = True
        
        self.fetch_parent_variables()
        
    def fetch_parent_variables(self):
        self.data = self.master_object.data
        self.rx = self.master_object.rx
        self.ry = self.master_object.ry
        self.s0 = self.master_object.s0
        self.s1 = self.master_object.s1
        
    def set_model_pipeline(self, pipeline_params, xtr, xte, ytr, yte):
        self.xtr = xtr
        self.xte = xte
        self.ytr = ytr
        self.yte = yte
        self.scaler = pipeline_params.get('scaler')
        self.compressor = pipeline_params.get('compressor')
        self.ann_model = pipeline_params.get('ann_model')

    def run_preprocessing(self):
        '''
        Runs preprocessing (only once).
        '''
        t0 = perf_counter()

        self.master_object.log_message(f'> Fitting {self.scaler.__class__.__name__} scaler...')
        self.xtr = self.scaler.fit_transform(self.xtr)
        self.xte = self.scaler.transform(self.xte)

        self.master_object.log_message(f'> Fitting {self.compressor.__class__.__name__} compressor...')
        self.xtr = self.compressor.fit_transform(self.xtr)
        self.xte = self.compressor.transform(self.xte)
        
        # self.compressor_to_fit = False
        
        self.master_object.log_message(
            '> Finished preprocessing! (Time elapsed: {:.2f} sec.)'.format(perf_counter()-t0))
    
    def get_batches(self, X, y, bs):
        shuffled_idx = np.arange(len(y))
        np.random.shuffle(shuffled_idx)
        for i in range(0, len(y), bs):
            batch_idx = shuffled_idx[i:i+bs]
            output = X[batch_idx]
            rgbs = y[batch_idx]            
            yield i, output, rgbs
    
    def train(self, training_params):
        optimizer = training_params.get('optimizer')
        bs = training_params.get('batch size')
        epochs = training_params.get('epochs')
        self.pbar.start()
        for epoch in range(epochs):
            self.master_object.log_message(f'> Training ANN model (epoch {epoch}/{epochs})...')
            for i, xtr_batch, ytr_batch in self.get_batches(self.xtr, self.ytr, bs):
                loss = self.train_batch(xtr_batch, ytr_batch, optimizer)
            valid_pred = self.ann_model(self.xte, training=False)
            valid_loss = self.loss_function(valid_pred, self.yte).numpy()
            self.valid_losses.append(valid_loss)
            self.master_object.log_message(
                '\n > Epoch #{} - Loss={:.3f} / {:.3f}'.format(epoch, loss, valid_loss))
            self.pbar.step(int(epoch/epochs*100))
            self.master.update_idletasks()
            self.total_epochs += 1
        self.pbar.stop()
        self.plot_loss(bs=bs)
        self.master_object.log_message('> Finished training! Loss={:.3f} / {:.3f}'.format(loss, valid_loss))
    
    def test(self, data=None):
        class_probas = self.ann_model(data, training=False)
        class_predis = tf.nn.softmax(class_probas)
        class_predis = tf.argmax(class_probas, axis=1, output_type=tf.int32)
        return class_predis.numpy()
    
    def train_batch(self, xtr_batch, ytr_batch, optimizer):
        with tf.GradientTape() as tape:
            model_output = self.ann_model(xtr_batch, training=False) # True with Dropout            
            train_loss = self.loss_function(model_output, ytr_batch)
        self.train_losses.append(train_loss.numpy())
        gradients = tape.gradient(train_loss, self.ann_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.ann_model.trainable_variables))
        return train_loss.numpy()
    
    def start_training(self, training_params):
        # if self.compressor_to_fit: # This causes problems... shut down for now.
        self.run_preprocessing()
        self.train(training_params)
    
    def predict_classes(self):
        '''
        Predicts classes, both per-pixel and per QR code square.
        '''
        blue = [0,104,255,128]
        orange = [255,175,0,128]
        black = [0, 0, 0, 255]
        white = [255, 255, 255, 255]
        
        self.fetch_parent_variables()
        dataflat = self.data.reshape((self.rx*self.ry, self.s0*self.s1))
        dataflat = self.scaler.transform(dataflat)
        dataflat = self.compressor.transform(dataflat)
        classes = self.test(dataflat).reshape((self.rx, self.ry)).astype(np.bool)
        pred = np.empty((self.rx, self.ry, 4), dtype=np.uint8)
        pred[classes] = blue
        pred[~classes] = orange
        
        rx, ry = classes.shape
        QR_output = np.zeros(pred.shape, dtype=np.uint8)
        
        for xs, xe, ys, ye in QR_window_generator(rx, ry, 
                                                  wsx=self.rx//29, 
                                                  wsy=self.ry//29, # How to determine that?
                                                  ):
            QR_output[xs:xe, ys:ye] = white if np.mean(classes[xs:xe, ys:ye]) > 0.5 else black
        
        return pred, QR_output
           
    def loss_function(self, predicted_target, real_target):
        predicted_target = tf.nn.softmax(predicted_target)
        real_target= tf.cast(real_target, tf.int32)
        real_target = tf.one_hot(real_target, 2)
        cce = tf.keras.losses.CategoricalCrossentropy()
        cross_entropy = cce(real_target, predicted_target)
        loss = tf.reduce_mean(cross_entropy)
        return loss
    
    def plot_loss(self, bs):
        fig, ax = plt.subplots(figsize=(3,3))
        # ax.plot(np.arange(len(self.train_losses)), self.train_losses, label='training', color='blue')
        ax.plot(np.arange(len(self.valid_losses))*bs, self.valid_losses, label='validation', color='orange')
        plt.tight_layout()
        plt.legend(loc=2)
        plt.close()
        loss_plot = FigureCanvasAgg(fig)
        s, (width, height) = loss_plot.print_to_buffer()
        loss_plot = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        self.host_canvas.show_rgba(loss_plot)
        self.master_object.nbk.select(1)