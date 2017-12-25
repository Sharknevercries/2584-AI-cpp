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
	rndenv(const std::string& args = "") : agent("name=rndenv role=environment " + args), alpha(0.0025f), active(true) {
		if (property.find("seed") != property.end())
			engine.seed(int(property["seed"]));
		if (property.find("active") != property.end())
			active = ((int)property["active"] == 1) ? true : false;
		if (active) {
			if (property.find("alpha") != property.end())
				alpha = float(property["alpha"]);

			tn = tuple_netwrok({
				{0, 1, 2, 3, 4},
				{0, 1, 4, 5, 6},
				{1, 2, 3, 5, 6},
				{5, 6, 7, 9, 10},
			});

			if (property.find("load") != property.end())
				tn.load_weights(property["load"]);
		}
	}

	~rndenv() {
		if (property.find("save") != property.end())
			tn.save_weights(property["save"]);
	}

	virtual void open_episode(const std::string& flag = "") {
		if (active) {
			episode.clear();
			episode.reserve(32768);
		}
	}

	virtual void close_episode(const std::string& flag = "") {
		if (active) {
			auto cur = episode.end() - 1, prev = cur - 1;
			tn.update(cur->before, alpha * (-tn.estimate(cur->before)));		
			while (cur != episode.begin()) {
				float td_error = cur->reward + tn.estimate(cur->before) - tn.estimate(prev->before);
				tn.update(prev->before, alpha * td_error);
				cur = prev;
				--prev;
			}
		}
	}

	virtual action take_action(const board& b) {
		std::uniform_int_distribution<int> popup(0, 3);
		const int tile = popup(engine) ? 1 : 3;

		if (active) {
			float max_value = -1e9;
			int best_pos = -1;
			board temp1(b), temp2(b);
			for (const int pos : space) {
				if (b(pos) != 0) continue;
				float value = 0;
				temp1(pos) = 1, temp2(pos) = 3;
				value += min_node(1, temp1) * 0.75;
				value += min_node(1, temp2) * 0.25;
				temp1(pos) = 0, temp2(pos) = 0;
				if (value > max_value) {
					max_value = value;
					best_pos = pos;
				}
			}

			episode.push_back(state(b, -1)); // step reward for env
			return best_pos == -1 ? action() : action::place(tile, best_pos);
		}
		else {
			std::shuffle(space.begin(), space.end(), engine);
			for (const int pos : space) {
				if (b(pos) != 0) continue;
				return action::place(tile, pos);
			}
		}

		return action();
	}
private:
	float min_node(const int level, const board& b) {
		float min_value = 1e9;
		for (int op = 0; op < 4; ++op) {
			board temp(b);
			int reward = action::move(op).apply(temp);
			float esti = 0;
			if (reward != -1) {
				if (level - 2 > 0)
					esti = max_node(level - 1, temp);
				else
					esti = tn.estimate(temp);
				min_value = std::min(min_value, esti);
			}
			else
				min_value = std::min(min_value, 0.0f);
		}
		return min_value;
	}

	float max_node(const int level, const board& b) {
		float max_value = -1e9;
		board temp1(b), temp2(b);
		for (const int pos : space) {
			if (b(pos) != 0) continue;
			float value = 0;
			temp1(pos) = 1, temp2(pos) = 3;
			value += min_node(level - 1, temp1) * 0.75;
			value += min_node(level - 1, temp2) * 0.25;
			temp1(pos) = 0, temp2(pos) = 0;
			max_value = std::max(max_value, value);
		}
		return max_value;
	}

private:

	struct state {
		state(const board& before, int reward) : before(before), reward(reward) {}
		board before;
		int reward;
	};

	std::default_random_engine engine;
	tuple_netwrok tn;
	std::vector<state> episode;
	float alpha;
	bool active;
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
			{0, 1, 4, 5, 6},
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
				// esti = tn.estimate(temp);
				esti = min_node(2, temp);
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
					esti = min_node(level - 1, temp);
				else
					esti = tn.estimate(temp);
			}
			max_value = std::max(max_value, reward + esti);
		}
		return max_value;
	}

	float min_node(int level, const board& b) {
		float min_value = 1e9;
		board temp1(b), temp2(b);
		for (int i = 0; i < 16; ++i) {
			if (b(i) != 0)	continue;
			float v = 0;
			temp1(i) = 1, temp2(i) = 3;
			v += max_node(level - 1, temp1) * 0.75;
			v += max_node(level - 1, temp2) * 0.25;
			temp1(i) = 0, temp2(i) = 0;
			min_value = std::min(min_value, v);
		}
		return min_value;
	}

private:
	tuple_netwrok tn;

	struct state {
		state() {}
		state(const board& after, const action& act, int reward) : after(after), move(act), reward(reward) {}
		board after;
		action move;
		int reward;
	};

	std::vector<state> episode;
	float alpha;

private:
	std::default_random_engine engine;
};
