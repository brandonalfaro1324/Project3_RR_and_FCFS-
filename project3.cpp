#include <stdio.h>      // printf use
#include <iostream>     // Get cout, endl
#include <unistd.h>     // POSIX operating system calls
#include <fstream>      // For getting Input/Output files
#include <string.h>     // Including String 
#include <vector>       // Using vectors to store Jobs

#include <queue>

using std::string;
using std::ifstream;
using std::ios;

using std::cout;
using std::endl;

using std::vector;

using std::queue;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////



////////////////////////
// Struct for Jobs being created
struct Jobs_{
    char job_id;
    int starting_time;
    int duration_time;

    // Variables that we may need.
    int finish_time;
    int turnaround_time;

};
////////////////////////



////////////////////////
// Class is used to create Jobs and push them in a Vector to keep track
class Job_Tracker{
    public:

    Job_Tracker(){
        total_jobs = 0;
    }

    // De-allocate Struct objects from Vector
    ~Job_Tracker(){

        // Loop trough until everything has been de-allocate
        for(int i = 0; i < stacking_jobs.size(); i++){
            
            Jobs_ *tmp_var = stacking_jobs[i];
            //cout << "DELETING DATA: " << tmp_var->job_id << " - Starting Time: " << tmp_var->starting_time << " - Duration Time: " << tmp_var->duration_time << endl;
            delete tmp_var;
        }
    }
    

    // add_job() Allocates jobs and pushes them in Vector
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
        
        total_jobs += 1;

        // Allocate "Jobs_" and store ID and numbers, then push to Vector
        Jobs_ *allocating_jobs = new Jobs_;
        allocating_jobs->job_id = job_id;
        allocating_jobs->starting_time = job_values[0];
        allocating_jobs->duration_time = job_values[1];

        stacking_jobs.push_back(allocating_jobs);
    }

    void Add_Tota_Time(int totaL_time){
        total_finish_time = totaL_time;
    }

    int Get_Tota_Time(){
        return total_finish_time;
    }


    // Current index of vector
    Jobs_ *Job_Service_Time(int vector_index){
        return stacking_jobs[vector_index];
    } 

    // Total number of jobs in vector
    int Get_Tota_Jobs(){
        return total_jobs;

    }

    // Variables needed for class
    private:
    int total_finish_time;
    int total_jobs;
    vector <Jobs_ *> stacking_jobs;
};
////////////////////////



////////////////////////
bool Check_Correct_Input(int, string, Job_Tracker *);
void FCFS(Job_Tracker *);
void FF(Job_Tracker *);
////////////////////////



////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////
int main(int argc, char* argv[]) { 


    Job_Tracker jobs_vector;

    // Collect Information
    string file_name = argv[1];

    if(Check_Correct_Input(argc, file_name, &jobs_vector) != true){
        cout << "EXITING EARLY..." << endl;
        return 0;
    }
    
    // -----Begin work here----
    FCFS(&jobs_vector);
    FF(&jobs_vector);

    return 0;
}
////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////
bool Check_Correct_Input(int total_inputs, string file_name_exist, Job_Tracker *jobs_vector){

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
            jobs_vector->add_job(string_element);
        }
        // Return true for file success
        file_sucess = true;
    }
    // return sucess of opening file
    return file_sucess;
}
////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////

void FCFS(Job_Tracker *jobs_){

    printf("\nFCFS\n\n");

    // Total Serive Time counter
    int current_service_timer = 0;

    // Loop by total job counts
    for(int i = 0; i < jobs_->Get_Tota_Jobs(); i++){
        
        // Get first element in job vector
        Jobs_ *current_job = jobs_->Job_Service_Time(i);

        // Print ID of job and print few the spaces needed to start X in below
        printf("%c", current_job->job_id);
        printf("%*c", current_service_timer+1, ' ');
        
        // Loop through the total number of Service Time and print X
        for(int j = 0; j < current_job->duration_time; j++){
            printf("X");
        }
        
        // Print new line and update "current_service_timer"
        printf("\n");
        current_service_timer += current_job->duration_time;
    }
    printf("\n");

    // Add total time finish to class.
    jobs_->Add_Tota_Time(current_service_timer);

}









////////////////////////


void FF(Job_Tracker *jobs_){

    Job_Tracker *tmp_job = jobs_;

    // Intialize queue to hold jobs
    queue<Jobs_ *> job_queue;
    
    // Intialize variables and array to print results
    int total_jobs = tmp_job->Get_Tota_Jobs();
    int total_time = tmp_job->Get_Tota_Time() + 2;

    char grid[total_jobs][total_time];

    // Push first Job to Queue
    int current_job_index = 0;
    job_queue.push(tmp_job->Job_Service_Time(current_job_index++));


    // Empty everything in array
    for(int i = 0; i < total_jobs; i++){
        grid[i][0] = 'A';
        grid[i][1] = 'B';
        for(int j = 2; j < total_time; j++){
            grid[i][j] = '0';
        }
        cout << endl;
    }




    for(int i = 2; i < total_time; i++){

        Jobs_ *current_job = job_queue.front();
        job_queue.pop();
       
        // Collect data
        // int array_location = 65 - current_job.job_id
        int array_column = 65 - (int) current_job->job_id;

        grid[array_column][i] = 'X';
        // Check next job to do work
        // 
        job_queue.push(current_job);
    }





    
    // Print statement to check results
    for(int i = 0; i < total_jobs; i++){        
        cout << grid[i][0];
        cout << grid[i][1];
        for(int j = 2; j < total_time; j++){
            cout << grid[i][j];
        }
        cout << endl;
    }
}



////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////