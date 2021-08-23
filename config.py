from imports import *

class Config:
    """
    Dieses Dokument enthält alle Zahlen.
    Im weiteren Code sollen keine magic Numbers vorkommen.
    Jeder Parameter, jede Kalibration wird hier definiert.
    """

    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Zeige in der Konsole nur Errormeldungen
    
    px_cm_ratio =  4                    # Maßstab  4 pixel = 1 cm          
    size_image_cm = 250                 # Seitenlänge des chunk-Quadrats
    line_nice_width_cm = 2              # Linenbreite der Fahrbahnränder
    center_distance_cm = 20             # rechter Abstand zur Mittellinie
    line_segmentated_width_cm = 4       # Breite der Fahrbahnränder im segmentierten Bild
    street_width_cm = 40                # Breite einer Fahrbahn
    drive_period_cm = 24                # Periodendauer der Fahrtpunkte
    dot_period_cm = 40                  # Periodendauer der gestrichelten Mittellinie
    box_offset = 40                     # Offset für Objektlabel Intersection
    dot_length_cm = 20                  # Pulsweite der gestrichelten Mittellinie

    file_list = ["line", "line", "line", "line"]    # Initialisiere die Strecke mit vier chunks
    degree_list = [0,0,0,0]
    disorder_list = [0,0,0,0]

    h_size_cm = 500                     # Größe des ROIs
    v_size_cm = 500 
    
    input_size_px = 128                 # Input Größe des neuronalen Netzes
    camera_resolution = (640,480)       # Auflösung der Fahrzeugkamera
    interrupted_lines = False           # Haben die Fahrbahnmarkierungen Lücken?
    magnitude = 50                      # Stärke der Störung auf dem Bild [0, 1]

    # Kameraparameter für die Perspektiventransformation
    camera_perspective = np.float32([[163.0, 289.0], [240.5, 250.5], [398.5, 250.6], [474.7, 289.6]])                                   
    bird_perspective = np.float32([[300 + 500, 1400.0 + 200], [300 + 500, 1000.0 + 200], [700 + 500, 1000.0 + 200], [700 + 500, 1400.0 + 200]])
    pre_transform_matrix = cv.getPerspectiveTransform(bird_perspective, camera_perspective)    # Vorwärts Transformationsmatrix   
    post_transform_matrix = cv.getPerspectiveTransform(camera_perspective, bird_perspective)    



    # Notwendige Umrechnungen der Einheiten
    size_image_px = int(size_image_cm * px_cm_ratio)
    line_nice_width_px = int(line_nice_width_cm * px_cm_ratio)
    center_distance_px = int(center_distance_cm * px_cm_ratio)
    line_segmentated_width_px = int(line_segmentated_width_cm * px_cm_ratio)
    street_width_px = int(street_width_cm * px_cm_ratio)
    drive_period_px = int(drive_period_cm * px_cm_ratio)
    dot_period_px = int(dot_period_cm * px_cm_ratio)
    dot_length_px = int(dot_length_cm * px_cm_ratio)    
    h_size_px = int(h_size_cm * px_cm_ratio) 
    v_size_px = int(v_size_cm * px_cm_ratio) 


    # Der Pfad der Log-Dateien
    ld='logs/training_results/train'
    # Ip und Port für Tensorboard
    h="127.0.0.1"
    p="6010" 


    # Parameter für das neuronale Netzwerk
    
    # Menge der Daten
    TRAINSET_SIZE = 100 # len(glob(train_path + "*.jpg"))
    VALSET_SIZE = 1000 # len(glob(val_path + "*.jpg"))
    # Anzahl Stichproben pro Batch
    BATCH_SIZE = 32
    # Datenformat der etikettierten Stichproben
    image_format = "*.jpg"
    amount_test_samples = 5
    
    STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
    VALIDATION_STEPS = VALSET_SIZE // BATCH_SIZE
    output_shapes = ([int(input_size_px/2), input_size_px, 1], [int(input_size_px/2), input_size_px, 1])
    # Anzahl der Farbkanäle
    N_CHANNELS = 1
    # Anzahl zu segmentierender Klassen
    N_CLASSES = 5
    # Zufälliger Schlüssel zum Mischen des Datensatzes
    SEED = 42

    # 
    BUFFER_SIZE = 100
    # Anzahl Epochen
    EPOCHS = 500
    # Pfade für etikettierte Stichproben
    train_path = "data/images/trainset/synthetic_set/"
    val_path = "data/images/valset/*/"
    
    # Wurde das Netzwerk bereits vortrainiert
    pretrain = False

    # Lade die Architektur und Parameter des Netzwerks.
    with open('./model.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    model.load_weights('./parameters.h5')
    model.compile()
    
    # Überdeckung des Fahrzeuges in der Fahrzeugkamera
    points = np.array([[int(input_size_px*0.65), int(input_size_px*0.5*0.65)], 
                    [int(input_size_px*0.35), int(input_size_px*0.5*0.65)],
                    [int(input_size_px*0.25), int(input_size_px*0.5)], 
                    [int(input_size_px*0.75), int(input_size_px*0.5)]])
