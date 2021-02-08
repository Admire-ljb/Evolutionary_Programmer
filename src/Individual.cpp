#include<iostream>
#include<Eigen/Dense>
#include<vector>
#include<ros.h>

class Individual
{
private:
	std::vector<float> waypoints;
	std::vector<float> trajectory;
	float fitness;
public:
	Individual(float start_point, float target, std::vector<float> &map, int coding_type=00) 
	{
		this->fitness = calFitness();
	}
	float calFitness();
};
