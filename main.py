import os
#import utils.create_dataset as create_dataset
import utils.create_dataset as create_dataset

class Main:
    def __init__(self):
        self.name = "Main Class"
        self.options = {
            '1': self.option1,
            '2': self.option2,
            '3': self.option3,
            '0': self.exit,
        }
        self.running = True
        
    def option1(self):
        # Código para crear dataset
        create_dataset.start_process()
        
    def option2(self):
        # Código para entrenar modelo
        a= True
        
    def option3(self):
        # Código para usar modelo
        a= True
        
    def exit(self):
        self.running = False

    def show_menu(self):
        print("\n")
        print("1. Crear dataset")
        print("2. Entrenar modelo")
        print("3. Usar modelo")
        print("0. Salir")

    def run(self):
        while self.running:
            self.show_menu()
            choice = input("Selecciona una opción: ")
            action = self.options.get(choice)
            if action:
                action()
            else:
                print("Opción no válida. Intenta de nuevo.")
                
if __name__ == "__main__":
    app = Main()
    app.run()