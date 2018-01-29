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
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

//using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  particles.resize(0);
  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;
    particles.emplace_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  std::default_random_engine gen;

  for (auto& p : particles) {
    // predict
    double theta_mod = p.theta + yaw_rate * delta_t;
    if (yaw_rate > 0.0001) {
      p.x += velocity / yaw_rate * (sin(theta_mod) - sin(p.theta));
      p.y += velocity / yaw_rate * (cos(p.theta) - cos(theta_mod));  
    } else {
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);  
    }
    p.theta = theta_mod;

    // add gaussian noise
    std::normal_distribution<double> dist_x(p.x, std_pos[0]);
    std::normal_distribution<double> dist_y(p.y, std_pos[1]);
    std::normal_distribution<double> dist_theta(p.theta, std_pos[2]);

    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);

    // normalize theta
    while (p.theta > 2.0 * M_PI)
      p.theta -= 2.0 * M_PI;
    while (p.theta < 0.0)
      p.theta += 2.0 * M_PI;  
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  for (auto& obs : observations) {
    double min_dist = std::numeric_limits<double>::max();
    for (auto pred : predicted) {
      double act_dist = dist(obs.x, obs.y, pred.x, pred.y);
      if (min_dist > act_dist) {
        min_dist = act_dist;
        obs.id = pred.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
    const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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

  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];

  for (auto& p : particles) {

    // keep landmarks within sensor range
    std::vector<LandmarkObs> predictions;
    for (auto lm : map_landmarks.landmark_list) {
      if (dist(p.x, p.y, lm.x_f, lm.y_f) <= sensor_range) {
        predictions.emplace_back(LandmarkObs{ lm.id_i, lm.x_f, lm.y_f });
      }
    }

    // create transformed observations from vehicle coords to map coords
    std::vector<LandmarkObs> tr_observations;
    for (auto obs : observations) {
      double tr_x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
      double tr_y = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;
      tr_observations.emplace_back(LandmarkObs{ obs.id, tr_x, tr_y });
    }

    dataAssociation(predictions, tr_observations);

    p.weight = 1;

    for (auto obs : tr_observations) {
      // get x,y coords of associated prediction
      double pr_x, pr_y;
      for (auto pred : predictions) {
        if (pred.id == obs.id) {
          pr_x = pred.x;
          pr_y = pred.y;
        }
      }

      // calculate observation weight using multivariate Gaussian
      double norm_term = 1 / (2 * M_PI * sig_x * sig_y);
      double exponent =  -1 * (pow(pr_x - obs.x, 2) / (2 * sig_x * sig_x) + (pow(pr_y - obs.y, 2) / (2 * sig_y * sig_y)));
      double obs_weight = norm_term * exp(exponent);

      // update particle weight
      p.weight *= obs_weight;
    }
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::vector<Particle> new_particles;

  // vector of all the current weights
  std::vector<double> weights;
  for (int i = 0; i < num_particles; ++i) {
    weights.push_back(particles[i].weight);
  }

  std::default_random_engine gen;
  std::discrete_distribution<> d(weights.begin(), weights.end());
  for (int i = 0; i < num_particles; ++i) {
    new_particles.emplace_back(particles[d(gen)]);
  }

  particles = new_particles; 
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                   const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
  return particle;
}

std::string ParticleFilter::getAssociations(Particle best)
{
  std::vector<int> v = best.associations;
  std::stringstream ss;
  copy( v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
std::string ParticleFilter::getSenseX(Particle best)
{
  std::vector<double> v = best.sense_x;
  std::stringstream ss;
  copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
std::string ParticleFilter::getSenseY(Particle best)
{
  std::vector<double> v = best.sense_y;
  std::stringstream ss;
  copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
