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

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  cout << "PF:init start" << endl; 

  num_particles = 10;

  // seed noise generation
  default_random_engine gen;
  normal_distribution<double> N_x_init(0, std[0]);
  normal_distribution<double> N_y_init(0, std[1]);
  normal_distribution<double> N_theta_init(0, std[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = x;
    p.y = y;
    p.theta = theta;
    p.weight = 1.0;

    // add noise
    p.x += N_x_init(gen);
    p.y += N_y_init(gen);
    p.theta += N_theta_init(gen);

    particles.push_back(p);
  }

  is_initialized = true;

  cout << "PF:init end" << endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  cout << "PF:prediction start" << endl;

  // seed noise generation
  default_random_engine gen;
  normal_distribution<double> N_x(0, std_pos[0]);
  normal_distribution<double> N_y(0, std_pos[1]);
  normal_distribution<double> N_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {

    // access ith particle
    Particle p = particles[i];

    // calculate new state
    if (yaw_rate < 0.001) {  
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    } 
    else {
      p.x     += velocity / yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
      p.y     += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
      p.theta += yaw_rate * delta_t;
    }

    // add noise
    p.x += N_x(gen);
    p.y += N_y(gen);
    p.theta += N_theta(gen);
  }

  cout << "PF:prediction end" << endl;

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.
  cout << "PF:dataAssociation start" << endl;

  for (unsigned int i = 0; i < observations.size(); i++) {
    
    // grab current observation
    LandmarkObs o = observations[i];

    // init minimum distance to maximum possible
    double min_dist = numeric_limits<double>::max();

    // init id of landmark from map placeholder to be associated with the observation
    int map_id = -1;
    
    for (unsigned int j = 0; j < predicted.size(); j++) {

      // grab current prediction
      LandmarkObs p = predicted[j];

      // get distance between current/predicted landmarks
      double cur_dist = dist(o.x, o.y, p.x, p.y);

      // find the predicted landmark nearest the current observed landmark
      if (cur_dist < min_dist) {

        min_dist = cur_dist;
        map_id = p.id;

      }

    }

    // set the observation's id to the nearest predicted landmark's id
    o.id = map_id;

  }
  
  cout << "PF:dataAssociation end" << endl;

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
    std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
  //   for the fact that the map's y-axis actually points downwards.)
  //   http://planning.cs.uiuc.edu/node99.html
  cout << "PF:updateWeights start" << endl;

  // for each particle...
  for (unsigned int i = 0; i < particles.size(); i++) {

    // get the particle x, y coordinates
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    // create a vector to hold the predicted map landmark locations from the particle's viewpoint
    vector<LandmarkObs> predictions;

    // for each map landmark...
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

      // get x,y coordinates
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;
      
      // only consider landmarks within sensor range of the particle (rather than using the "dist" method considering a circular 
      // region around the particle, this considers a rectangular region but is computationally faster)
      if (fabs(lm_x - p_x) <= sensor_range && fabs(lm_y - p_y) <= sensor_range) {

        LandmarkObs prediction;

        // transform the landmark location to what the particle would observe
        // rotation and translation:
        prediction.x = lm_x * cos(-p_theta) - lm_y * sin(-p_theta) - p_x;
        prediction.y = lm_x * sin(-p_theta) + lm_y * cos(-p_theta) - p_y;
        prediction.id = lm_id;

        // add prediction to vector
        predictions.push_back(prediction);

      }

    }

    // perform dataAssociation for the predictions
    dataAssociation(predictions, observations);
  }



  cout << "PF:updateWeights end" << endl;
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  cout << "PF:resample start" << endl;



  cout << "PF:resample end" << endl;
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
  
