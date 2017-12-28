#pragma once
#include "../agent.h"

class dummy_player : public agent {
public:
	dummy_player(const std::string& args = "") : agent("role=player " + args) {
		if (property.find("seed") != property.end())
			engine.seed(int(property["seed"]));
	}

	action take_action(const board& before) {
		int opcode[] = { 0, 1, 2, 3 };
		std::shuffle(opcode, opcode + 4, engine);
		for (int op : opcode) {
			board b = before;
			if (b.move(op) != -1) return action::move(op);
		}
		return action();
	}

private:
	std::default_random_engine engine;
};
