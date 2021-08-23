from imports import *

class Dataset:
    """
    Die Klasse Dataset enthält Funktionen um Datensätze zu erstellen, bearbeiten und visualisieren.
    """

    @tf.function
    def parse_image(img_path):
        """
        Diese Funktion lädt eine etikettierte Stichprobe aus dem Speicher.
        Input:
        img_path (Pfad zum echten Bild der Straße. Der Pfad enthält ein Wildcard für den Namen des Bildes)
        Output:
        {image, mask} (etikettierte Stichprobe)
        """
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=1)

        # Bestimme den Pfad der Etikette
        mask_path = tf.strings.regex_replace(img_path, "images", "annotations")
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_jpeg(mask, channels=1)       
             
        # Normierung der Daten
        image = tf.cast(image, tf.float32)/255.0
        mask = tf.round(tf.cast(mask, tf.float32)/50.0)
        mask = tf.clip_by_value(mask, clip_value_min = 0, clip_value_max = 4)

        return  image, mask
       
      
    def display_sample(sample_image, sample_mask, pred_mask):
        """
        Diese Klasse visualisiert eine Inferenz des neuronalen Netzes.
        Input: 
        sample_image, sample_mask, prediction_mask
        """

        #print(sample_image.shape)
        #print(sample_mask.shape)  
        
        plt.figure(figsize=(18, 18))
        #plt.axis('off') 
        plt.subplot(1, 3, 1)
        plt.title('Input Image')
        array_image = np.array(sample_image) 
        plt.imshow(array_image, cmap='gray', vmin=0, vmax=1)
     
        plt.subplot(1, 3, 2)
        plt.title('True Mask')                              
        array_mask = np.array(sample_mask)
        normed_mask = np.rint(array_mask)           
        plt.imshow(normed_mask, cmap='gray', vmin=0, vmax=4)
        
        # Prüfe ob Prediction angezeigt werden soll
        if np.array(pred_mask).any() != None:
            plt.subplot(1, 3, 3)
            plt.title('Prediction')                              
            array_pred = np.array(pred_mask)           
            normed_pred = np.rint(array_pred)                       
            plt.imshow(normed_pred, cmap='gray', vmin=0, vmax=4)
                                 
        plt.show(block=False)
        plt.pause(3)
        plt.close()



    def create_mask(pred_mask):
        """
        Die Funktion wandelt den Ausgang des neuronalen Netzes in ein lesbares 2d-Bild um.
        Input:
        pred_mask [input_size_px, input_size_px/2, N_CLASSES] (tf.Tensor Für jeden Pixel besteht ein Vektor mit der Wahrscheinlichkeit jeder Klassenzugehörigkeit)
        Output:
        pred_mask [input_size_px, input_size_px/2, 1] (tf.Tensor Anstelle des Vektors, steht für jedes Pixel die wahrscheinlichste Klasse da)
        """

        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = tf.expand_dims(pred_mask, axis=-1)

        return pred_mask
                             
        
    def create_dataset(path, image_format):
        """
        Erzeuge einen Datensatz aus den Daten an dem übergebenen Pfad.      
        
        
        Die Funktion ist fehlerhaft und geht von der ehemaligen Struktur der Trainingsdaten aus.
        Muss geändert werden. Siehe
        """
        AUTOTUNE = tf.data.experimental.AUTOTUNE
            
        dataset = tf.data.Dataset.list_files(path + image_format)    
        dataset = dataset.map(Dataset.parse_image, num_parallel_calls=AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=Config.BUFFER_SIZE, seed=Config.SEED)
        dataset = dataset.repeat()
        dataset = dataset.batch(Config.BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        
        return dataset
        
     
    def create_hybrid(src_path):
        """
        Diese Funktion erstellt aus den realen Trainingsdaten hybride Stichproben.
        Dabei werden die Bilder und Etiketten leicht abgeändert.
        Die Stichproben werden im Dateiensystem abgespeichert.
        Input:
        src_path (Pfad zu den realen Daten, die abgeändert werden sollen)      
        """
        index = 0
        # src_path = 'data/annotations/trainset/set*/*.jpg'

        for i in range(10):
            for mask_path in glob.glob(src_path):

                image_path = re.sub(r"annotations", "images", mask_path)
                
                
                mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)  
                image = cv.imread(image_path, cv.IMREAD_GRAYSCALE) 
               
               

                if random.choice([True, False]):

                    row,col = image.shape

                    # Wähle ein zufälliges Obstacle aus
                    number = random.randint(1,34)
                    obstacle = cv.imread(f"white_box/box_{number}.jpg", cv.IMREAD_GRAYSCALE) # 
                    
                    # Wähle eine zufällige Größe aus
                    height = random.randint(20,40)
                    width = random.randint(30,35)

                    obstacle = cv.resize(obstacle, (width, height), interpolation = cv.INTER_LINEAR)

                    # Wähle einen zufälligen Platz für das Overlay
                    x_place = random.randint(0, int(row*0.7) - height)
                    y_place = random.randint(0, int(col*0.7) - width)

                    overlay_mask = obstacle * 0
                    overlay_mask[obstacle > 100] = 1
                    image_mask = 1 - overlay_mask
                    
                    image[x_place : x_place + height, y_place : y_place + width] = overlay_mask * obstacle + image_mask * image[x_place : x_place + height, y_place : y_place + width]
                    
                    
                    space = mask[x_place : x_place + height, y_place : y_place + width]
                    space[obstacle > 50] = 0
                    mask[x_place : x_place + height, y_place : y_place + width] = space  
               
                else: 
                    image = Visualization.add_overlay(image)        
               
                image = Visualization.add_noise(image, random.randint(0, 30)) 
                  

                
                # Abdeckung des Autos mit einer schwarzen Fläche
                image = cv.polylines(image, [Config.points], True, 0, 1)
                image = cv.fillPoly(image, [Config.points], 0)
                
                cv.imwrite(f'data/annotations/trainset/hybrid_set/image_{index}.jpg', mask)
                cv.imwrite(f'data/images/trainset/hybrid_set/image_{index}.jpg', image)
                
                index = index + 1
        
