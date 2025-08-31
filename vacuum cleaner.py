class VacuumCleaner:
    def __init__(self, shape, advantage):
        self.shape = shape
        self.advantage = advantage
        self.status = "idle"

    def show_advantage(self):
        print(f" {self.shape} Vacuum Advantage: {self.advantage}")

    def command(self, action):
        if action.lower() == "start":
            self.status = "cleaning"
            print(f" {self.shape} vacuum started cleaning...")
        elif action.lower() == "stop":
            self.status = "stopped"
            print(f" {self.shape} vacuum stopped.")
        elif action.lower() == "left":
            print(f"{self.shape} vacuum turned left.")
        elif action.lower() == "right":
            print(f" {self.shape} vacuum turned right.")
        elif action.lower() == "dock":
            self.status = "charging"
            print(f" {self.shape} vacuum returning to dock for charging.")
        else:
            print(" Unknown command.")


vacuums = {
    "circle": VacuumCleaner("Circle", "Best for smooth mobility and avoiding obstacles"),
    "square": VacuumCleaner("Square", "Better at reaching into corners"),
    "triangle": VacuumCleaner("Triangle", "Sharp tip cleans deep corners"),
    "rectangle": VacuumCleaner("Rectangle", "Efficient for large open spaces")
}

print(" Smart Vacuum Cleaner Simulation")
print("Available shapes: circle, square, triangle, rectangle")
print("Commands: start, stop, left, right, dock, exit")

while True:
    shape_choice = input("\nChoose a vacuum shape: ").lower()
    if shape_choice == "exit":
        print(" Simulation ended.")
        break
    if shape_choice not in vacuums:
        print(" Invalid shape! Try again.")
        continue

    vacuums[shape_choice].show_advantage()

    action = input(f" Enter command for {shape_choice} vacuum: ").lower()
    if action == "exit":
        print(" Simulation ended.")
        break
    
    vacuums[shape_choice].command(action)
