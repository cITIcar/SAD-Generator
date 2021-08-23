from imports import *
from main import road, visualization

class Train:
    """
    Die Klasse Train enthält Funktionen, die für das Training der Gewichte erforderlich sind.
    
    """
  
        
    def our_generator():
        """
        Diese Funktion übernimmt bereitet die etikettierte Stichprobe für
        das neuronale Netz vor.
        Vorsicht: Anstatt return wird yield verwendet. 
        Dadurch wird kein Platz im Speicher für die Bilder allokiert.
        Das ist effizienter, schließlich braucht man die Stichprobe nur ein Mal.       
        """
        while True: 
            camera_nice, camera_segment = Visualization.one_datapoint(visualization, road)
           
            camera_nice = camera_nice[ ..., tf.newaxis]
            camera_segment = camera_segment[ ..., tf.newaxis]
        
            #image = tf.image.resize(camera_nice, (int(Config.input_size_px/2), Config.input_size_px))
            #mask = tf.image.resize(camera_segment, (int(Config.input_size_px/2), Config.input_size_px))          
           
            mask = tf.reshape(camera_segment, [int(Config.input_size_px/2), Config.input_size_px, 1]) 
            image = tf.reshape(camera_nice, [int(Config.input_size_px/2), Config.input_size_px, 1])    
 
            image = tf.cast(image, tf.float32)/255.0
            mask = tf.round(tf.cast(mask, tf.float32)/ 50.0)  
            
            yield image,mask


    def train_net(train_dataset, val_dataset):
        """
        Diese Funktion trainiert die Parameter.
        Diese Callbacks bestehen:
        - DisplayCallback(): Nach jeder Epoche wird die Inferenz geprüft. 
            Damit lässt sich der Trainingsverlauf visualisieren.
        - Tensorboard_callback: Der Trainingsverlauf wird geloggt und kann später im Browser angezeigt wwerden.
        - EarlyStopping: Wenn 10 Epochen lang keine Verbesserung eintritt, bricht das Training ab.
        - ModelCheckpoint: Parameterveränderungen, die das Netz verbessern, werden direkt gespeichert.
        run_eagerly=True ist dafür da, das .numpy() bei Tensoren funktioniert
        """

        logdir = os.path.join("logs", "training_results")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        callbacks = [
            DisplayCallback(),
            tensorboard_callback,
            tf.keras.callbacks.EarlyStopping(patience=15, verbose=1),
            tf.keras.callbacks.ModelCheckpoint('parameters.h5', verbose=1, save_best_only=True, save_weights_only=True)
        ]
        

        """
        Learnrate Callback, um zu reduzieren
        verschiedene Optimizer probieren
        Batch Normalisation
        """
        #loss = Train.weighted_categorical_crossentropy([0.35, 0.1, 0.1, 0.35, 0.1])
        #optimizer = tf.keras.optimizers.Adam(lr=0.001)
        
        optimizer=tfa.optimizers.RectifiedAdam(lr=1e-3)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

        Config.model.compile(optimizer=optimizer, loss = loss,
                          metrics=['accuracy'], run_eagerly=True)


        history = Config.model.fit(train_dataset, 
                            epochs=Config.EPOCHS,
                            steps_per_epoch=Config.STEPS_PER_EPOCH,
                            validation_steps=Config.VALIDATION_STEPS,
                            validation_data=val_dataset,
                            callbacks=callbacks
                            )
        return history


class DisplayCallback(tf.keras.callbacks.Callback):
    """
    Die Klasse enthält Callback-Funktionen.
    Diese können im Zuge des Trainings ausgeführt werden.
    """
    def on_epoch_end(self, epoch, logs=None):
        """
        Diese Funktion wird nach jeder Epoche ausgeführt.
        Sie visualisiert den Trainingsstand der Epoche.
        Dabei werden Soll- und Istzustand einer Stichprobe verglichen.
        """
        
        clear_output(wait=True)
        
        
        #Config.model.load_weights(os.path.join('parameters.h5'))

        #optimizer=tfa.optimizers.RectifiedAdam(lr=1e-3)
        #loss = tf.keras.losses.SparseCategoricalCrossentropy()
        
        #Config.model.compile(optimizer=optimizer, loss = loss, metrics=['accuracy'])
                
        val_dataset = Dataset.create_dataset("data/images/valset/*/", Config.image_format)
            
        for image, mask in val_dataset.take(1):           
            image = image[0][tf.newaxis, ...]
            
        inference = Config.model.predict(image)   
        pred_mask = Dataset.create_mask(inference)
        Dataset.display_sample(image[0], mask[0], pred_mask[0])
        
        array_mask = np.array(pred_mask[0])
        flatten_mask = array_mask.flatten()
        set_mask = set(flatten_mask)
        print("Classes Predicted: " + str(set_mask))
                  
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

