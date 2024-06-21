#pragma once
#include <chrono>
#include <iomanip>
#include <thread>
#include <iostream>

class Timer
{
public:

	Timer()
	{
		start = std::chrono::high_resolution_clock::now();
	}
	~Timer(){

	}

	std::chrono::duration<double> End() {
		end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration = end - start;
		//std::cout << std::setprecision(15) << duration.count() << std::endl << std::setprecision(3);
		return duration;
	}
private:
	std::chrono::time_point<std::chrono::steady_clock> start, end;
};

