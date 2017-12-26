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
	rndenv(const std::string& args = "") : agent("name=vanilla role=environment " + args), enable_evil(true) {
		if (property.find("seed") != property.end())
			engine.seed(int(property["seed"]));
		if (property.find("evil") != property.end())
			enable_evil = ((int)property["evil"] == 1) ? true : false;
		if (enable_evil) {
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

	virtual action take_action(const board& b) {
		std::uniform_int_distribution<int> popup(0, 3);
		const int tile = popup(engine) ? 1 : 3;

		if (enable_evil) {
			float m = 1e9, m4 = m * 4;
			int best_pos = -1;
			int level = 1;
			const int empty_tiles = b.empty_tile_count();

			if (empty_tiles < 2)
				level = 5;
			else if (empty_tiles < 5)
				level = 3;

			board temp1(b), temp2(b);
			for (const int pos : space) {
				if (b(pos) != 0) continue;
				float v1, v2, ev;
				temp1(pos) = 1, temp2(pos) = 3;
				v1 = max_node(level, temp1, -1e9, 1e9);
				v2 = max_node(level, temp2, -1e9, m4 - 3 * v1);
				ev = v1 * 0.75 + v2 * 0.25;
				temp1(pos) = 0, temp2(pos) = 0;
				if (m > ev) {
					m = ev;
					m4 = m * 4;
					best_pos = pos;
				}
			}
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
	float max_node(const int level, const board& b, float alpha, float beta) {
		float m = alpha;
		bool has_child = false;
		for (int op = 0; op < 4; ++op) {
			board temp(b);
			int reward = action::move(op).apply(temp);
			float esti = 0;
			if (reward != -1) {
				has_child = true;
				if (level - 1 > 0)
					esti = min_node(level - 1, temp, m - reward, beta);
				else
					esti = tn.estimate(temp);
				m = std::max(m, reward + esti);
				if (m >= beta)
					return m;
			}
		}
		return has_child ? m : 0;
	}

	float min_node(const int level, const board& b, float alpha, float beta) {
		const float a4 = alpha * 4;
		float m = beta, m4 = 4 * m;
		bool has_child = false;
		board temp1(b), temp2(b);
		for (int i = 0; i < 16; ++i) {
			if (b(i) != 0)	continue;
			has_child = true;

			float v1 = 0, v2, v3, ev;
			temp1(i) = 1, temp2(i) = 3;
			v1 = max_node(level - 1, temp1, -1e9, 1e9);
			v3 = v1 * 3;
			v2 = max_node(level - 1, temp2, a4 - v3, m4 - v3);
			temp1(i) = 0, temp2(i) = 0;
			ev = v1 * 0.75 + v2 * 0.25;
			if (m > ev) {
				m = ev;
				m4 = 4 * m;
			}
			if (m <= alpha)
				return m;
		}
		return has_child ? m : 0;
	}

private:

	std::default_random_engine engine;
	tuple_netwrok tn;
	bool enable_evil;
};

/**
 * TODO: player (non-implement)
 * always return an illegal action
 */
class player : public agent {
public:
	player(const std::string& args = "") : agent("name=chocolate role=player " + args), alpha(0.0025f), enable_search(true), enable_player(true) {
		episode.reserve(32768);
		if (property.find("seed") != property.end())
			engine.seed(int(property["seed"]));
		if (property.find("alpha") != property.end())
			alpha = float(property["alpha"]);
		if (property.find("player") != property.end())
			enable_player = ((int)property["player"] == 1) ? true : false;
		if (property.find("search") != property.end())
			enable_search = ((int)property["search"] == 1) ? true : false;

		if (enable_player) {
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
	~player() {
		if (enable_player) {
			if (property.find("save") != property.end())
				tn.save_weights(property["save"]);
		}
	}

	virtual void open_episode(const std::string& flag = "") {
		if (enable_player) {
			episode.clear();
			episode.reserve(32768);
		}
	}

	virtual void close_episode(const std::string& flag = "") {
		if (enable_player) {
			auto cur = episode.end() - 1, prev = cur - 1;
			tn.update(cur->after, alpha * (-tn.estimate(cur->after)));		
			while (cur != episode.begin()) {
				float td_error = cur->reward + tn.estimate(cur->after) - tn.estimate(prev->after);
				tn.update(prev->after, alpha * td_error);
				cur = prev;
				--prev;
			}
		}
	}

	virtual action take_action(const board& before) {
		if (enable_player) {
			state best(before, action(), 0);
			float best_value = -1e9;

			for (int op = 0; op < 4; ++op) {
				auto act = action::move(op);
				auto temp = board(before);
				int reward = act.apply(temp);
				const int empty_tiles = temp.empty_tile_count();
				int level = 0;

				if (enable_search) {
					if (empty_tiles < 2)
						level = 6;
					else if (empty_tiles < 3)
						level = 4;
					else
						level = 2;
				}

				if (reward != -1) {
					float esti = 0;
					if (level > 0)
						esti = min_node(level, temp, best_value - reward, 1e9);
					else
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
		else {
			int opcode[] = { 0, 1, 2, 3 };
			std::shuffle(opcode, opcode + 4, engine);
			for (int op : opcode) {
				board b = before;
				if (b.move(op) != -1) return action::move(op);
			}
			return action();
		}
	}
private:
	float max_node(const int level, const board& b, float alpha, float beta) {
		float m = alpha;
		bool has_child = false;
		for (int op = 0; op < 4; ++op) {
			board temp(b);
			int reward = action::move(op).apply(temp);
			float esti = 0;
			if (reward != -1) {
				has_child = true;
				if (level - 1 > 0)
					esti = min_node(level - 1, temp, m - reward, beta);
				else
					esti = tn.estimate(temp);
				m = std::max(m, reward + esti);
				if (m >= beta)
					return m;
			}
		}
		return has_child ? m : 0;
	}

	float min_node(int level, const board& b, float alpha, float beta) {
		const float a4 = alpha * 4;
		float m = beta, m4 = 4 * m;
		bool has_child = false;
		board temp1(b), temp2(b);
		for (int i = 0; i < 16; ++i) {
			if (b(i) != 0)	continue;
			has_child = true;

			float v1, v2, v3, ev;
			temp1(i) = 1, temp2(i) = 3;
			v1 = max_node(level - 1, temp1, -1e9, 1e9);
			v3 = v1 * 3;
			v2 = max_node(level - 1, temp2, a4 - v3, m4 - v3);
			temp1(i) = 0, temp2(i) = 0;
			ev = v1 * 0.75 + v2 * 0.25;
			if (m > ev) {
				m = ev;
				m4 = 4 * m;
			}
			if (m <= alpha)
				return m;
		}
		return has_child ? m : 0;
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
	bool enable_search;
	bool enable_player;

private:
	std::default_random_engine engine;
};
