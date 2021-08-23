from imports import *

"""

xavier tensor units fp16 Beachten!

Read Me Schreiben

"""


road = Road()
visualization = Visualization()


def main(args):
    """
    Wähle ein Programm:
    - show_generator:           Datengenerator visualisieren
    - train:                    Training des Netzes
    - test:                     Ist- und Sollzustand der Inferenz prüfen
    - debug:                    Kompilieren der Tensorboard-Visualisierung
    - create_elements:          Erzeugung der Straßenelemente
    
    """

    print(args.program)  
    commands[args.program]()


def generator():      
    """
    Diese Funktion führt den Datengenerator aus. Es werden die etikettierten Stichproben
    unter dem Pfad "path" gespeichert.
    """
    while True:                        
        path = Config.val_path
        camera_nice, camera_segment = Visualization.one_datapoint(visualization, road)                                
        Visualization.make_metadata(visualization, camera_nice, camera_segment, path)  
    


def train():             
    """
    Diese Funktion erstellt einen Validierungs- (val_dataset) und Trainingsdatensatz (train_dataset) 
    und trainiert damit das neuronale Netzwerk. Die Architektur des Netzwerks wird aus der Datei "model.json" bezogen.
    Falls vortrainierte Parameter verwendet werden sollen, kann die Datei "parameters.h5" importiert werden.
    """
    
    # Lade die Architektur und Parameter des Netzwerks.
    with open(os.path.join('model.json'), 'r') as json_file:
        loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    if Config.pretrain == True:
        model.load_weights(os.path.join('parameters.h5'))

    val_dataset = Dataset.create_dataset(Config.val_path, Config.image_format)  # "data/images/valset/*/"
    train_dataset = Dataset.create_dataset(Config.train_path, Config.image_format)  # "data/images/trainset/synthetic_set/"

    history = Train.train_net(train_dataset, val_dataset)
    Explain.plot_learning(history)

        
def test():   
    """
    Berechnet die Metrik aus der Arbeit:
    Güte der Inlier und Verfügbarkeit der Inlier
    """
    Test.line_accuracy() 


def debug():           
    """
    Diese Funktion startet Tensorboard und kompiliert die Log-Datei zu einer visualisierbaren Darstellung
    """
    
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', Config.ld, '--host', Config.h, '--port', Config.p])
    tb.main()


def create_elements():     
    """
    Diese Funktion generiert die Straßenelemente für den Datengenerator
    Sie muss nur ausgeführt werden, wenn die Elemente geändert wurden.    
    """

    line = Line("line", "chunks")
    right_curve = Curve("curve", "chunks", "right")
    left_curve = Curve("curve", "chunks", "left")
    intersection = Intersection("intersection", "chunks")
    Line.create_chunk(line)
    Curve.create_chunk(right_curve)
    Curve.create_chunk(left_curve)
    Intersection.create_chunk(intersection)
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--program",
        nargs="?",
        type=str,

        default="show_generator",

        help="Programm to execute",
    )


    commands = {
        'generator': generator,
        'train': train,
        'test': test,
        'debug': debug,
        'create_elements': create_elements,                                          
    }
  
    
    args = parser.parse_args()

    main(args)
