#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include "weight.h"
#include "env.h"
#include "feature.h"
#include "tuple_network.h"

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			property[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string name() const { return property.at("name"); }
	virtual std::string role() const { return property.at("role"); }
	virtual void notify(const std::string& msg) { property[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> property;
};

/**
 * evil (environment agent)
 * add a new random tile on board, or do nothing if the board is full
 * 1-tile: 75%
 * 3-tile: 25%
 */
class rndenv : public agent {
public:
	rndenv(const std::string& args = "") : agent("name=rndenv role=environment " + args) {
		if (property.find("seed") != property.end())
			engine.seed(int(property["seed"]));
	}

	virtual action take_action(const board& after) {
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;
			std::uniform_int_distribution<int> popup(0, 3);
			int tile = popup(engine) ? 1 : 3;
			return action::place(tile, pos);
		}
		return action();
	}

private:
	std::default_random_engine engine;
};

/**
 * TODO: player (non-implement)
 * always return an illegal action
 */
class player : public agent {
public:
	player(const std::string& args = "") : agent("name=player role=player " + args), alpha(0.0025f) {
		episode.reserve(32768);
		if (property.find("seed") != property.end())
			engine.seed(int(property["seed"]));
		if (property.find("alpha") != property.end())
			alpha = float(property["alpha"]);

		tn = tuple_netwrok({
			{0, 1, 2, 3, 4},
			{1, 2, 3, 5, 6},
			{5, 6, 7, 9, 10},
		});

		if (property.find("load") != property.end())
			tn.load_weights(property["load"]);
	}
	~player() {
		if (property.find("save") != property.end())
			tn.save_weights(property["save"]);
	}

	virtual void open_episode(const std::string& flag = "") {
		episode.clear();
		episode.reserve(32768);
	}

	virtual void close_episode(const std::string& flag = "") {
		auto cur = episode.end() - 1, prev = cur - 1;
		tn.update(cur->after, alpha * (-tn.estimate(cur->after)));		
		while (cur != episode.begin()) {
			float td_error = cur->reward + tn.estimate(cur->after) - tn.estimate(prev->after);
			tn.update(prev->after, alpha * td_error);
			cur = prev;
			--prev;
		}
	}

	virtual action take_action(const board& before) {
		state best(before, action(), 0);
		float best_value = -1e9;

		for (int op = 0; op < 4; ++op) {
			auto act = action::move(op);
			auto temp = board(before);
			int reward = act.apply(temp);
			// int empty_tiles = temp.empty_tile_count();
			if (reward != -1) {
				float esti = 0;
				// if (empty_tiles < 2)
				// 	esti = expeceted_node(6, temp);
				// else if (empty_tiles < 3)
				// 	esti = expeceted_node(4, temp);
				// else if (empty_tiles < 4)
				// 	esti = expeceted_node(2, temp);
				// else
				esti = tn.estimate(temp);
				if (reward + esti > best_value) {
					best = state(temp, act, reward);
					best_value = reward + esti;
				}
			}
		}
		episode.push_back(best);

		return best.move;
	}
private:
	float max_node(const int level, const board& b) {
		float max_value = 0;
		for (int op = 0; op < 4; ++op) {
			board temp(b);
			int reward = action::move(op).apply(temp);
			float esti = 0;
			if (reward != -1) {
				if (level - 1 > 0)
					esti = expeceted_node(level - 1, temp);
				else
					esti = tn.estimate(temp);
			}
			max_value = std::max(max_value, reward + esti);
		}
		return max_value;
	}

	float expeceted_node(int level, const board& b) {
		int empty_tiles = b.empty_tile_count();
		float v = 0;
		board temp1(b), temp2(b);
		for (int i = 0; i < 16; ++i) {
			if (b(i) != 0)	continue;
			temp1(i) = 1, temp2(i) = 3;
			v += max_node(level - 1, temp1) / empty_tiles * 0.75;
			v += max_node(level - 1, temp2) / empty_tiles * 0.25;
		}
		return v;
	}

private:
	tuple_netwrok tn;

	struct state {
		state() {}
		state(const board& after, const action& act, int reward) : after(after), move(act), reward(reward) {}
		// TODO: select the necessary components of a state
		board after;
		action move;
		int reward;
	};

	std::vector<state> episode;
	float alpha;

private:
	std::default_random_engine engine;
};
