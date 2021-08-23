from imports import *


class Test:
    """
    Die Klasse Test stellt Funktionen zum Testen eines neuronalen Netzes zur Verfügung.  
    """
       

    def calculate_metrics(sample_mask, pred_mask):
        """
        Beschreibung:
            Diese Funktion berechnet die Metriken einer einzelnen Stichprobe.   
        Input: 
            sample_mask (Ground Truth: Etikette der Stichprobe)
            pred_mask (Prediction: Inferenz des neuronalen Netzwerks)      
        Output: 
            IoU_classwise (Intersection over Union jeder einzelnen Klasse)
            accuarcy_classwise (Accuracy jeder einzelnen Klasse)
            total_accuracy (Durchschnitt der Accuracy über alle Klassen)
            total_IoU (Durchschnitt von IoU über alle Klassen                   
        """
        
        # Umwandlung von Tensor zu Array       
        array_pred = np.array(pred_mask)  
        array_mask = np.array(sample_mask)  
        
        # Casten der segmentierten Pixelwerte in int
        array_pred = np.rint(array_pred)          
        array_mask = np.rint(array_mask)         
   
        if False: 
            # Debugging: Menge der Klassen, die gefunden werden
            flatten_mask = array_pred.flatten()
            set_mask = set(flatten_mask)
            print(set_mask)
        
        
        accuarcy_classwise = []
        IoU_classwise = []       
        # ones = np.ones((int(Config.input_size_px/2), Config.input_size_px, Config.N_CHANNELS))
        
        # Anzahl Pixel im gesamten Bild
        total_pixels = Config.input_size_px**2/2
        
        # Metriken werden für jede Klasse einzeln berechnet
        for class_num in range(Config.N_CLASSES):
        
            # Arrays der Pixel die in der Maske oder Inferenz der Klasse x zugeordnet sind
            mask_positives = class_num == array_mask
            pred_positives = class_num == array_pred
                   
            # Array der Pixel die in der Etikette und der Inferenz der Klasse x zugeordnet worden sind
            true_positives = mask_positives & pred_positives
            true_negatives = ~mask_positives & ~pred_positives
            intersection = mask_positives | pred_positives
                    
            # Anzahl der richtig der Klasse x zugeordneten Pixel (true positives)
            amount_true_positives = np.sum(true_positives)           
            amount_true_negatives = np.sum(true_negatives)
            amount_intersection = np.sum(intersection)
                   
            class_accuracy = (amount_true_positives + amount_true_negatives)/total_pixels
            class_IoU = amount_true_positives/amount_intersection if amount_intersection != 0 else 0
            
            accuarcy_classwise.append(round(class_accuracy, 4))
            IoU_classwise.append(round(class_IoU, 4))
        
        # Array der Pixel die der richtigen Klasse zugeordnet wurden
        total_accuracy = round(sum(accuarcy_classwise)/Config.N_CLASSES, 4)
        total_IoU = round(sum(IoU_classwise)/Config.N_CLASSES, 4)

        return IoU_classwise, accuarcy_classwise, total_accuracy, total_IoU

        
    def test_net(val_dataset): # path
        """
        Diese Funktion berechnet den Mittelwert der Metriken über einen größeren Datensatz aus.
        Input: 
            val_dataset (Testdaten)
        Output: 
            Konsolenausgabe der Metriken: IoU, Accuracy, Inferenzdauer, Speicherbedarf 
        """
        np.set_printoptions(threshold=sys.maxsize)
        # Kompilierung des Modells
        with open(os.path.join('model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        model.load_weights(os.path.join('parameters.h5'))

        optimizer=tfa.optimizers.RectifiedAdam(lr=1e-3)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

        model.compile(optimizer=optimizer, loss = loss,
                          metrics=['accuracy'])

        # Initialisierung der Listen zum Sammeln der Daten
        IoU_classwise_list = []
        accuarcy_classwise_list = []
        total_accuracy_list = []
        total_IoU_list = []
        inference_time_list = []
        memory_used_list = []
        
        # Bestimme den gesamten GPU-Speicher
        gpu_memory_total = nvgpu.gpu_info()[0]["mem_total"]
       
        # Größe des Testdatensatzes
        amount_test_samples = Config.amount_test_samples
        
        # Der erste Predict ist komisch und wird hier unberücksichtigt
        for image, mask in val_dataset.take(1):
            sample_image, sample_mask = image, mask

        one_img_batch = sample_image[0][tf.newaxis, ...]         
        inference = Config.model.predict(one_img_batch)
        
        # Berechne die Metriken über den Testdatensatz
        for x in range(amount_test_samples):
            for image, mask in val_dataset.take(1):
                sample_image, sample_mask = image, mask
             
            one_img_batch = sample_image[0][tf.newaxis, ...]    
            
            start_time = time.time()
            inference = Config.model.predict(one_img_batch)
            end_time = time.time()
            inference_time = end_time - start_time
            
            #print(inference.shape)
            pred_mask = Dataset.create_mask(inference)    
            #print(pred_mask.shape)
            
            IoU_classwise, accuarcy_classwise, total_accuracy, total_IoU = Test.calculate_metrics(sample_mask[0], pred_mask[0])         
            #Dataset.display_sample([sample_image[0], sample_mask[0], pred_mask[0]])
                       
            gpu_memory_used = nvgpu.gpu_info()[0]["mem_used"]
            memory_used_list.append(gpu_memory_used)           
            IoU_classwise_list.append(IoU_classwise)
            accuarcy_classwise_list.append(accuarcy_classwise)
            total_accuracy_list.append(total_accuracy)
            total_IoU_list.append(total_IoU)
            inference_time_list.append(inference_time)

            test = np.array(inference[0])
            test = test[:,:,2]
            
            if True:                
                
                prediction = Dataset.create_mask(inference[0])
            
                #Dataset.display_sample(image[0], mask[0], test)
                        
                plt.figure(figsize=(18, 18))
                #plt.axis('off') 
                plt.subplot(1, 3, 1)
                plt.title('Input Image')
                array_image = np.array(sample_image[0]) 
                plt.imshow(array_image, cmap='gray', vmin=0, vmax=1)
             
                plt.subplot(1, 3, 2)
                plt.title('True Mask')                              
                array_mask = np.array(sample_mask[0])
                normed_mask = np.rint(array_mask)           
                plt.imshow(normed_mask, cmap='gray', vmin=0, vmax=4)
                
                plt.subplot(1, 3, 3)
                plt.title('Prediction')                                                    
                plt.imshow(prediction, cmap='gray', vmin=0, vmax=4)        
        
                plt.show()
        
                array_mask = np.array(prediction)
                flatten_mask = array_mask.flatten()
                set_mask = set(flatten_mask)
                #print(set_mask)

                      
        # Berechne den Mittelwert der Metriken
        IoU_classwise_average = np.round(np.sum(np.array(IoU_classwise_list), axis = 0)/amount_test_samples, 4)
        total_IoU_average = round(sum(total_IoU_list)/amount_test_samples, 4)
        accuarcy_classwise_average = np.round(np.sum(np.array(accuarcy_classwise_list), axis=0)/amount_test_samples, 4)
        total_accuracy_average = round(sum(total_accuracy_list)/amount_test_samples, 4)
        inference_time_average = round(sum(inference_time_list)/amount_test_samples, 4)
        memory_used_average = round(sum(memory_used_list)/amount_test_samples, 4)
        memory_used_percentage = round(memory_used_average/ gpu_memory_total * 100, 2)

        # Konsolenausgabe der Ergebnisse
        print("IoU per Class: " + str(IoU_classwise_average) + ", IoU total: " + str(total_IoU_average))
        print("Accuracy per Class: " + str(accuarcy_classwise_average) + ", Accuracy total: " + str(total_accuracy_average))
        print("Inference Time [s]: " + str(inference_time_average))
        print("Memory Used [MB]: " + str(memory_used_average) + ", Memory Used [%]: " + str(memory_used_percentage))

        """
        Die Berechnung des Durchschnitts ist bei Klassen, die nicht in jedem Bild vorkommen nicht korrekt.
        Es darf nur so oft geteilt werden, wie die KLasse vorkam.            
        """
        
        
    def line_accuracy():
        """
        Diese Funktion berechtet eine Metrik zur Quantifizierung er false positives und false negatives der lokalisierten Kanten.
        Es wird nicht nur die Genauigkeit der Segmentierung, sondern auch die der Bildverarbeitung berücksichtigt.
        Es wird die gefundene kritische Kante in die Heatmap gelegt und der Fehlerabstand quantifiziert.
               
        """
        
        index = 5000     
        for pred_path in glob.glob('data/lines/valset/hybrid_set/pred_*.jpg'): 
    
            index += 1
            line_path = re.sub(r"pred", "image", pred_path)
            
            image_path = re.sub(r"lines", "images", line_path)
            pred = cv.imread(pred_path, cv.IMREAD_GRAYSCALE)
            image = cv.imread(image_path)
            
            heatmap = cv.imread(line_path, cv.IMREAD_GRAYSCALE) 
            
            sum = np.sum(heatmap[pred > 200])/255
            amount = np.size(heatmap[pred > 200])
            gewichteter_fehlabstand = sum / amount
            

            count_v = np.sum(np.any(heatmap > 250, axis=1))
            test_v = np.sum(np.any(np.bitwise_and((heatmap > 250), (pred>200)), axis=1))

            count_h = np.sum(np.any(heatmap > 250, axis=0))
            test_h = np.sum(np.any(np.bitwise_and((heatmap > 250), (pred>200)), axis=0))
            
            verfügbarkeit = (test_v + test_h)/(count_v + count_h)


            print("----------Neue Stichprobe: " + str(index) + " ------------------")
            
            print("Güte der Inlier: " + str(gewichteter_fehlabstand))
            print("Verfügbarkeit der Inlier: " + str(verfügbarkeit))        
            

            
            heatmap = cv.cvtColor(heatmap,cv.COLOR_GRAY2RGB)
            image[:,:,2] = cv.add(image[:,:,2],pred)            
            heatmap[:,:,2] = cv.add(heatmap[:,:,2],pred)
            
            image[:,:,0][pred > 200] = 0   
            image[:,:,1][pred > 200] = 0   
            
            heatmap[:,:,0][pred > 200] = 0 
            heatmap[:,:,1][pred > 200] = 0            
            
            # Debugging
            if True:
                combined = np.concatenate((heatmap, image), axis=1)    
                combined = cv.resize(combined, (1500, 500))    
                cv.imshow("test", combined)
                cv.imwrite("images/image_"+str(index)+".jpg", combined)
                cv.waitKey(0) 
                cv.destroyAllWindows() 
            
