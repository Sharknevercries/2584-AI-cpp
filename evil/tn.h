#pragma once
#include "../agent.h"

class tn_evil : public agent {
public:
	tn_evil(const std::string& args = "") : agent("role=environment " + args), alpha(0.0025f), enable_search(false), threshold(-1), prev_max_tile(-1), epsilon(0.0) {
		if (property.find("seed") != property.end())
			engine.seed(int(property["seed"]));
		if (property.find("search") != property.end())
			enable_search = ((int)property["search"] == 1) ? true : false;
		if (property.find("threshold") != property.end())
			threshold = (int)property["threshold"];
		if (property.find("alpha") != property.end())
			alpha = float(property["alpha"]);
		if (property.find("epsilon") != property.end())
			epsilon = float(property["epsilon"]);
		tn = tuple_network({
			{0, 1, 2, 3, 4},
			{0, 1, 4, 5, 6},
			{1, 2, 3, 5, 6},
			{5, 6, 7, 9, 10},
		});
		if (property.find("load") != property.end())
			tn.load_weights(property["load"]);
	}

	~tn_evil() {
		if (property.find("save") != property.end())
			tn.save_weights(property["save"]);
	}

	virtual void open_episode(const std::string& flag = "") {
		prev_max_tile = -1;
		episode.clear();
		episode.reserve(32768);
	}

	virtual void close_episode(const std::string& flag = "") {
		if (episode.empty())
			return;
		auto cur = episode.end() - 1, prev = cur - 1;
		tn.update(cur->before, alpha * (-tn.estimate(cur->before)));		
		while (cur != episode.begin()) {
			float td_error = cur->reward + tn.estimate(cur->before) - tn.estimate(prev->before);
			tn.update(prev->before, alpha * td_error);
			cur = prev;
			--prev;
		}
	}

	virtual action take_action(const board& b) {
		const int max_tile = b.get_max_tile();
		action taken_action;

		if (threshold < 0 && prev_max_tile < max_tile) {
			switch_tuple_network(max_tile);
			prev_max_tile = max_tile;
		}
		
		if (max_tile >= threshold)
			taken_action = take_epsilon_greedy(b);
		else
			taken_action = take_random_move(b);
		episode.push_back(state(b, -1)); // step reward for env		
		return taken_action;
	}

private:
	float min_node(const int level, const board& b, float alpha, float beta) {
		float m = beta;
		bool has_child = false;
		for (int op = 0; op < 4; ++op) {
			board temp(b);
			int reward = action::move(op).apply(temp);
			float esti = 0;
			if (reward != -1) {
				has_child = true;
				if (level - 1 > 0)
					esti = max_node(level - 1, temp, alpha, m);
				else
					esti = tn.estimate(temp);
				m = std::min(m, esti);
				if (m <= alpha)
					return m;
			}
		}
		return has_child ? m : 0;
	}

	float max_node(const int level, const board& b, float alpha, float beta) {
		const float b4 = beta * 4;
		float m = alpha, m4 = m * 4;
		bool has_child = false;
		board temp1(b), temp2(b);
		for (const int pos : space) {
			if (b(pos) != 0) continue;
			has_child = true;

			float v1, v2, v3, ev;
			temp1(pos) = 1, temp2(pos) = 3;
			v1 = min_node(level - 1, temp1, -1e9, 1e9);
			v3 = v1 * 3;
			v2 = min_node(level - 1, temp2, m4 - v3 + 4, b4 - v3 + 4);
			temp1(pos) = 0, temp2(pos) = 0;
			ev = v1 * 0.75 + v2 * 0.25 - 1;
			if (ev > m) {
				m = ev;
				m4 = m * 4;
			}
			if (m >= beta)
				return m;
		}
		return has_child ? m : 0;
	}

	void switch_tuple_network(const int threshold) {
		const std::vector<int> available_network = {0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
		if(std::find(available_network.begin(), available_network.end(), threshold) != available_network.end()) {
			std::string weight_filename = "weights/ew_" + std::to_string(threshold);
			tn.load_weights(weight_filename);
		}
	}

	int get_search_level(const board& b) const {
		const int empty_tiles = b.get_empty_tile_count();
		if (empty_tiles < 3)
			return 7;
		else if (empty_tiles < 9)
			return 5;
		else
			return 3; 
	}

	action take_random_move(const board& b) {
		std::uniform_int_distribution<int> popup(0, 3);
		const int tile = popup(engine) ? 1 : 3;
		std::shuffle(space.begin(), space.end(), engine);
		for (const int pos : space) {
			if (b(pos) != 0) continue;
			return action::place(tile, pos);
		}
		return action();
	}
	
	action take_best_move(const board& b) {
		std::uniform_int_distribution<int> popup(0, 3);
		bool given_tile = false;
		int tile = 0;
		if (property.find("tile") != property.end()) {
			tile = (int)property["tile"];
			given_tile = true;
		}
		else
			tile = popup(engine) ? 1 : 3;
		
		int level = 1;
		if (enable_search)
			level = get_search_level(b);

		float max_value = -1e9, m4 = max_value * 4;
		int best_pos = -1;
		board temp1(b), temp2(b);
		for (const int pos : space) {
			if (b(pos) != 0) continue;
			float v1, v2, v3, ev;
			if (given_tile) {
				temp1(pos) = tile;
				ev = min_node(level, temp1, max_value, 1e9);
				temp1(pos) = 0;
			}
			else {
				temp1(pos) = 1, temp2(pos) = 3;
				v1 = min_node(level, temp1, -1e9, 1e9);
				v3 = v1 * 3;
				v2 = min_node(level, temp2, m4 - v3, 1e9);
				temp1(pos) = 0, temp2(pos) = 0;
				ev = v1 * 0.75 + v2 * 0.25;
			}			
			if (ev > max_value) {
				max_value = ev;
				m4 = max_value * 4;
				best_pos = pos;
			}
		}

		return best_pos == -1 ? action() : action::place(tile, best_pos);
	}

	action take_epsilon_greedy(const board& b) {
		std::uniform_real_distribution<> popup(0.0, 1.0);
		bool use_best_move = popup(engine) > epsilon;

		if (use_best_move)
			return take_best_move(b);
		else
			return take_random_move(b);
	}


private:
	struct state {
		state(const board& before, int reward) : before(before), reward(reward) {}
		board before;
		int reward;
	};

	std::default_random_engine engine;
	tuple_network tn;
	std::vector<state> episode;
	float alpha;
	bool enable_search;
	int threshold;
	int prev_max_tile;
	float epsilon;
};
