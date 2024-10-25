from search import romania_map
import SimpleProblemSolvingAgent as SPSA

def main():
    while True:  
        ## Print out all possible cities
        print("Here are all the possible Romania cities that can be traveled:")
        locations = romania_map.locations.keys()
        print(list(locations))

        # Ask for user input to get origin and destination cities
        while True:
            start = input("Please enter the origin city: ")
            if start not in locations:
                print(f"Could not find {start}, please try again.")
                continue  # ask again if the origin is invalid

            end = input("Please enter the destination city: ")
            if end not in locations:
                print(f"Could not find {end}, please try again.")
                continue  # ask for both origin and destination if the destination is invalid

            if start == end:
                print("The same city can't be both origin and destination. Please try again.")
                continue  # ask for both again if the origin and destination are the same
            
            # If inputs are valid break out of the loop and continue
            break

        # Run search algorithms inside of SimpleProblemSolvingAgent
        problem = SPSA.GraphProblem(start, end, romania_map)
        SPSA.runSearchAlgorithms(problem)
        
        ## Prompt if user wants to find another path, only accepts "yes" to continue
        again = input("Would you like to find the best path between two other cities? (yes/no): ")
        if again.lower() != "yes":
            print("Thank you for using our app!")
            break

if __name__ == '__main__':
    main()
