#pragma once
#include "../agent.h"

class random_evil : public agent {
public:
	random_evil(const std::string& args = "") : agent("role=environment " + args) {
		if (property.find("seed") != property.end())
			engine.seed(int(property["seed"]));
	}

	action take_action(const board& b) {
		std::uniform_int_distribution<int> popup(0, 3);
		const int tile = popup(engine) ? 1 : 3;
		std::shuffle(space.begin(), space.end(), engine);
		for (const int pos : space) {
			if (b(pos) != 0) continue;
			return action::place(tile, pos);
		}
		return action();
	}

private:
	std::default_random_engine engine;
};