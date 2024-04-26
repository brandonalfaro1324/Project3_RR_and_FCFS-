#include <stdio.h>      // printf use
#include <iostream>     // Get cout, endl
#include <unistd.h>     // POSIX operating system calls
#include <fstream>      // For getting Input/Output files
#include <string.h>     // Including String 
#include <queue>        // Using Queue to keep track of Jobs

using std::string;
using std::ifstream;
using std::ios;

using std::cout;
using std::endl;

using std::queue;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////

////////////////////////
bool Check_Correct_Input(int, string, Job_Tracker);
////////////////////////


////////////////////////
// Struct for Jobs being created
struct Jobs_{
    char job_id;
    int starting_time;
    int duration_time;
};
////////////////////////


////////////////////////
// Class is used to create Jobs and push them in a Queue to keep track
class Job_Tracker{
    public:
    
    // add_job() Allocates jobs and pushes them in Queue
    void add_job(string data_coming_in){

        // Create variable to store Job's ID 
        char job_id;

        // Create variables to store numbers from string
        int job_values[2];
        string number = "";
        int j = 0;

        // Loop trought String and extract Data
        for(int i = 0; i < data_coming_in.length() ; i++){

            // Extract Character
            if(isalpha(data_coming_in[i])){
                job_id = data_coming_in[i]; 
            }
            // Extract Number
            else if(isdigit(data_coming_in[i])){
                number += data_coming_in[i];

                // If the chracter after this one is a number, then 
                // there might be more than 1 digit
                while(isdigit(data_coming_in[i+1])){
                    number += data_coming_in[++i];
                }

                // Once number string is collected, make it integer
                job_values[j++] = stoi(number); 
                number = "";
            }
        }
        
        // Allocate "Jobs_" and store ID and numbers, then push to Queue
        Jobs_ *allocating_jobs = new Jobs_;
        allocating_jobs->job_id = job_id;
        allocating_jobs->starting_time = job_values[0];
        allocating_jobs->duration_time = job_values[1];

        stacking_jobs.push(allocating_jobs);
    }

    // De-allocate Struct objects from Queue
    ~Job_Tracker(){

        // Loop trough until everything has been de-allocated
        while(stacking_jobs.size() != 0){
            Jobs_ *tmp_var = stacking_jobs.front();

            cout << "DELETING DATA: " << tmp_var->job_id << " - " << tmp_var->starting_time << " - " << tmp_var->duration_time << endl;
            delete tmp_var;
            stacking_jobs.pop();
        }
    }

    private:
    queue <Jobs_ *> stacking_jobs;
};
////////////////////////

////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////
int main(int argc, char* argv[]) { 


    Job_Tracker jobs;

    // Collect Information
    string file_name = argv[1];

    if(Check_Correct_Input(argc, file_name, jobs) != true){
        return 0;
    }
    
    // -----Begin work here----


    return 0;
}
////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////
bool Check_Correct_Input(int total_inputs, string file_name_exist, Job_Tracker jobs){

    // Return success signal back to main
    bool file_sucess = false;

    // Open file
    ifstream new_file;
    new_file.open(file_name_exist, ios :: in);
    
    // Check if file exist
    // If false, return false back to main
    if(new_file.is_open()){

        // File does exist, extract data from files line by line
        string string_element;
        while(getline(new_file, string_element)){
            jobs.add_job(string_element);
        }
        // Return true for file success
        file_sucess = true;
    }
    // return sucess of opening file
    return file_sucess;
}
////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////