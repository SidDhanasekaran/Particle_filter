/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "particle_filter.h"
#include "map.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    num_particles = 200;
    for(int i=0; i< num_particles; i++){
        random_device rd;
        default_random_engine gen(rd());
        normal_distribution<double> gps_error_x(x, std[0]);
        normal_distribution<double> gps_error_y(y, std[1]);
        normal_distribution<double> gps_error_theta(theta, std[2]);

        double particle_x = gps_error_x(gen);
        double particle_y = gps_error_y(gen);
        double particle_theta = gps_error_theta(gen);
        double particle_weight = 1.0;

        Particle new_particle = {i, particle_x, particle_y, particle_theta, particle_weight};
        particles.push_back(new_particle);
        weights.push_back(particle_weight);
    }
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    for(int i=0; i< num_particles; i++){
        if(yaw_rate == 0){
            particles[i].x = particles[i].x + (velocity * delta_t) * cos(particles[i].theta);
            particles[i].y = particles[i].y + (velocity * delta_t) * sin(particles[i].theta);
        }else{
            particles[i].x = particles[i].x + (velocity/yaw_rate)*(sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
            particles[i].y = particles[i].y + (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
            particles[i].theta = particles[i].theta + (yaw_rate * delta_t);
        }


        random_device rd;
        default_random_engine gen(rd());
        normal_distribution<double> pos_error_x(particles[i].x, std_pos[0]);
        normal_distribution<double> pos_error_y(particles[i].y, std_pos[1]);
        normal_distribution<double> pos_error_theta(particles[i].theta, std_pos[2]);

        particles[i].x = pos_error_x(gen);
        particles[i].y = pos_error_y(gen);
        particles[i].theta = pos_error_theta(gen);
    }
	
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    for(int obs=0; obs<observations.size(); obs++){
        double obs_x = observations[obs].x;
        double obs_y = observations[obs].y;

        double temp_delta_l = 0.0;
        bool temp_delta_l_initialized = false;

        for(int l=0; l<predicted.size(); l++){
            double delta_x = obs_x - predicted[l].x;
            double delta_y = obs_y - predicted[l].y;

            double delta_l = sqrt(pow(delta_x, 2.0) + pow(delta_y, 2.0));

            if((!temp_delta_l_initialized) || (temp_delta_l > delta_l)) {
                temp_delta_l = delta_l;
                temp_delta_l_initialized = true;
                observations[obs].id = l;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks) {
    for(int i=0; i<num_particles; i++){
        double current_x = particles[i].x;
        double current_y = particles[i].y;
        double current_theta = particles[i].theta;

        vector<LandmarkObs> predicted_landmarks;
        for(int l=0; l<map_landmarks.landmark_list.size(); l++){
            int l_id = map_landmarks.landmark_list[l].id_i;
            double l_x = map_landmarks.landmark_list[l].x_f;
            double l_y = map_landmarks.landmark_list[l].y_f;

            double delta_x = l_x - current_x;
            double delta_y = l_y - current_y;

            double distance = sqrt(pow(delta_x, 2.0) + pow(delta_y, 2.0));
            if(distance<=sensor_range){
                l_x = delta_x * cos(current_theta) + delta_y * sin(current_theta);
                l_y = delta_y * cos(current_theta) - delta_x * sin(current_theta);
                LandmarkObs landmark_in_range = {l_id, l_x, l_y};
                predicted_landmarks.push_back(landmark_in_range);
            }
        }

        dataAssociation(predicted_landmarks, observations);

        double new_weight = 1.0;
        for(int obs=0; obs<observations.size(); obs++) {
            int l_id = observations[obs].id;
            double obs_x = observations[obs].x;
            double obs_y = observations[obs].y;

            double delta_x = obs_x - predicted_landmarks[l_id].x;
            double delta_y = obs_y - predicted_landmarks[l_id].y;

            double numerator = exp(- 0.5 * (pow(delta_x,2.0)*std_landmark[0] + pow(delta_y,2.0)*std_landmark[1] ));
            double denominator = sqrt(2.0 * M_PI * std_landmark[0] * std_landmark[1]);
            new_weight = new_weight * numerator/denominator;
        }
        weights[i] = new_weight;
        particles[i].weight = new_weight;

    }
	
}

void ParticleFilter::resample() {
    std::vector<Particle> resampled_particles;

    random_device rd;
    default_random_engine gen(rd());

    for(int counter=0; counter<particles.size(); counter++){
        discrete_distribution<int> index(weights.begin(), weights.end());
        resampled_particles.push_back(particles[index(gen)]);
    }

    particles = resampled_particles;
	
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
