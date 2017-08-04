/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#define _USE_MATH_DEFINES

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>
#include <cmath>

#include "particle_filter.h"

using namespace std;

static default_random_engine dre;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 1000;
	particles.clear();
	particles.resize(num_particles);
	weights.clear();
	weights.resize(num_particles, 1.0);

	normal_distribution<double> ndist_x(0.0, std[0]);
	normal_distribution<double> ndist_y(0.0, std[1]);
	normal_distribution<double> ndist_theta(0.0, std[2]);

	for (Particle &p : particles) {
		p.id = 0;
		p.x = x + ndist_x(dre);
		p.y = y + ndist_y(dre);
		p.theta = theta + ndist_theta(dre);
		p.weight = 1.0;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	double std_x = std_pos[0];
  	double std_y = std_pos[1];
  	double std_theta = std_pos[2];
	double meas_yaw = yaw_rate * delta_t;
	random_device rd;
	mt19937 dre(rd());

	for (Particle &p : particles) {
		double yaw = p.theta;

		if (fabs(yaw_rate) > 0.001) {
			p.x += velocity / yaw_rate * (sin(yaw + meas_yaw) - sin(yaw));
			p.y += velocity / yaw_rate * (cos(yaw) - cos(yaw + meas_yaw));
			p.theta += meas_yaw;
		} else {
			p.x += velocity * delta_t * cos(yaw);
			p.y += velocity * delta_t * sin(yaw);
		}

		normal_distribution<double> ndist_x(p.x, std_x);
		normal_distribution<double> ndist_y(p.y, std_y);
		normal_distribution<double> ndist_theta(p.theta, std_theta);

		p.x = ndist_x(dre);
    	p.y = ndist_y(dre);
    	p.theta = ndist_theta(dre);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (auto prediction : predicted) {
		double closest_dist = numeric_limits<double>::max();
		LandmarkObs* prev_closest_observation = NULL;

		for (LandmarkObs &observation : observations) {
			if (observation.id > 0) {
				continue;
			}

			double d = dist(observation.x, observation.y, prediction.x, prediction.y);
			
			if (d < closest_dist) {
				if (prev_closest_observation != NULL) {
					prev_closest_observation->id = -1;
				}
				
				observation.id = prediction.id;
				prev_closest_observation = &observation;
				closest_dist = d;
			}
		}
	}

	observations.erase(
    	remove_if(observations.begin(), observations.end(), [] (const LandmarkObs &lo) {
			return lo.id < 1;
		}), observations.end()
	);
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
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double sigma_x = std_landmark[0];
  	double sigma_y = std_landmark[1];

	for (Particle &p : particles) {
		vector<LandmarkObs> observations_map(observations);
		
		for (LandmarkObs &o : observations_map) {
			double vx = o.x * cos(p.theta) - o.y * sin(p.theta) + p.x;
			double vy = o.x * sin(p.theta) + o.y * cos(p.theta) + p.y;
		
			o.x = vx;
			o.y = vy;
			o.id = 0;
		}

		vector<LandmarkObs> predicted;
		LandmarkObs lo;
		
		for (Map::single_landmark_s landmark : map_landmarks.landmark_list) {
			if (dist(p.x, p.y, landmark.x_f, landmark.y_f) < sensor_range) {
				lo = {landmark.id_i, landmark.x_f, landmark.y_f};
				predicted.push_back(lo);
			}
		}
		
		dataAssociation(predicted, observations_map);

		map<int, LandmarkObs> predictedMap;

		for (LandmarkObs prediction : predicted) {
			predictedMap.insert({ prediction.id, prediction });
		}

		double weight_product = 1;
		
		for (LandmarkObs measurement : observations_map) {
			LandmarkObs predicted_measurement = predictedMap[measurement.id];
			double mu_x = predicted_measurement.x;
			double mu_y = predicted_measurement.y;
			double x = measurement.x;
			double y = measurement.y;
			double c1 = 1.0 / (2.0 * M_PI * sigma_x * sigma_y);
			double c2 = pow(x - mu_x, 2) / pow(sigma_x, 2);
			double c3 = pow(y - mu_y, 2) / pow(sigma_y, 2);
			double weight = c1 * exp(-0.5 * (c2 + c3));

			if (weight < 0.0001) {
				weight = 0.0001;
			}

			weight_product *= weight;
		}

		p.weight = weight_product;
  	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	double beta = 0.0;
  	random_device rd;
  	mt19937 genindex(rd());
  	uniform_int_distribution<int> dis(0, num_particles - 1);
  	int index = dis(genindex);
	vector<Particle> resampled_particles;
	double max_weight = 0.0;

  	resampled_particles.reserve(num_particles);
	weights.clear();
  	weights.reserve(num_particles);

  	for (Particle particle : particles) {
    	if (particle.weight > max_weight) {
      		max_weight = particle.weight;
		}

    	weights.push_back(particle.weight);
  	}

	mt19937 gen(rd());
  	uniform_real_distribution<double> dis_real(0, 2.0 * max_weight);

  	for (Particle particle : particles) {
    	beta += dis_real(gen);
    
		while (weights[index] < beta) {
      		beta -= weights[index];
      		index = (index + 1) % num_particles;
    	}
    
		resampled_particles.push_back(particles[index]);
  	}

	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
