#include <stdio.h>      // printf use
#include <iostream>     // Get cout, endl
#include <unistd.h>     // POSIX operating system calls
#include <fstream>      // For getting Input/Output files
#include <string.h>     // Including String 
#include <vector>       // Using vectors to store Jobs
#include <queue>        // Using queue for RR

using std::string;
using std::ifstream;
using std::ios;

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
    int total_process_time;
};
////////////////////////

////////////////////////
// Class is used to create Jobs and push them in a Vector to keep track
class Job_Tracker{
    
    public:
    Job_Tracker(){
        total_jobs = 0;
    }
//////

    // De-allocate Struct objects from Vector
    ~Job_Tracker(){

        // Loop trough until everything has been de-allocate
        for(int i = 0; i < stacking_jobs.size(); i++){
            Jobs_ *tmp_var = stacking_jobs[i];
            delete tmp_var;
        }
    }
//////

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
        allocating_jobs->total_process_time = 0;

        stacking_jobs.push_back(allocating_jobs);
    }
//////
    
    void Add_Tota_Time(int totaL_time){
        total_finish_time = totaL_time;
    }
//////
    
    int Get_Tota_Time(){
        return total_finish_time;
    }
//////

    // Current index of vector
    Jobs_ *Job_Service_Time(int vector_index){
        return stacking_jobs[vector_index];
    } 
//////

    // Total number of jobs in vector
    int Get_Tota_Jobs(){
        return total_jobs;

    }
//////

    // Private Variables needed for class
    private:
    int total_finish_time;
    int total_jobs;
    vector <Jobs_ *> stacking_jobs;
};
////////////////////////

////////////////////////
// Support functions
bool Check_Correct_Input(int, string, Job_Tracker *);

// Scheduling Algorithms
void FCFS(Job_Tracker *);
void FF(Job_Tracker *);
////////////////////////
////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////
int main(int argc, char* argv[]) { 

    // Intialize Object
    Job_Tracker jobs_vector;

    // Quit program is there is no 2 arguments
    if(argc != 2){
        printf("ERROR, NEEDS TWO ARGUMENTS...\n");
        return 0;
    }
    
    // Collect Information
    string file_name = argv[1];

    if(Check_Correct_Input(argc, file_name, &jobs_vector) != true){
        printf("ERROR, FILE DOES NOT EXIST...\n");
        return 0;
    }

    // Begin Program
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

    // Create temp "Job_Tracker" onject to hold incoming class
    Job_Tracker *tmp_job = jobs_;

    // Intialize queue to hold jobs
    queue<Jobs_ *> job_queue;
    
    // Intialize variables and array to print results
    int total_jobs = tmp_job->Get_Tota_Jobs();
    int total_time = tmp_job->Get_Tota_Time() + 2;

    // Intialize 2D array
    char grid[total_jobs][total_time];

    // Push first Job to Queue to begin process
    int current_job_index = 0;
    job_queue.push(tmp_job->Job_Service_Time(current_job_index++));

    // Add in Jod ID in first column of array and assign " " to each cell
    for(int i = 0; i < total_jobs; i++){
        grid[i][0] = tmp_job->Job_Service_Time(i)->job_id;
        grid[i][1] = ' ';

        // Loop through row and assign " "
        for(int j = 2; j < total_time; j++){
            grid[i][j] = ' ';
        }
    }

    // Skip column 1 and 2, Assign 'X' to other columns 
    for(int i = 2; i < total_time; i++){

        // Get first Job from Queue and pop it
        Jobs_ *current_job = job_queue.front();
        job_queue.pop();

        // Assign 'X' to correct row and column
        int array_column = (int) current_job->job_id - 65;
        grid[array_column][i] = 'X';

        // Intialize pointer and check if next Job exists
        Jobs_ *check_next_job;

        // If there are Jobs available, retrieve them
        if(current_job_index < total_jobs){
            check_next_job = tmp_job->Job_Service_Time(current_job_index);
        }

        // If the next job starting time is now, push next job 
        // to Queue before current job is pushed
        if(check_next_job->starting_time == i - 1){
            job_queue.push(check_next_job);
            current_job_index++;
        }

        // If current Job time has not pass over its duration, push it to Queue
        if(current_job->total_process_time++ < current_job->duration_time - 1){
            job_queue.push(current_job);
        }
    }

    // Print Array.    
    printf("\nFF\n\n");
    for(int i = 0; i < total_jobs; i++){        
        printf("%c", grid[i][0]);
        printf("%c", grid[i][1]);

        for(int j = 2; j < total_time; j++){
            printf("%c", grid[i][j]);
        }
        printf("\n");
    }
}
////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////