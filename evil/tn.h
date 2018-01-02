#pragma once
#include "../agent.h"

class tn_evil : public agent {
public:
	tn_evil(const std::string& args = "") : agent("role=environment " + args), alpha(0.0025f), enable_search(false), threshold(-1), prev_max_tile(-1) {
		if (property.find("seed") != property.end())
			engine.seed(int(property["seed"]));
		if (property.find("search") != property.end())
			enable_search = ((int)property["search"] == 1) ? true : false;
		if (property.find("threshold") != property.end())
			threshold = (int)property["threshold"];
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
		std::uniform_int_distribution<int> popup(0, 3);
		const int tile = popup(engine) ? 1 : 3;
		const int max_tile = b.get_max_tile();

		if (threshold < 0 && prev_max_tile < max_tile) {
			switch_tuple_network(max_tile);
			prev_max_tile = max_tile;
		}

		if (max_tile >= threshold) {
			int level = 1;

			if (enable_search)
				level = get_search_level(b);

			float max_value = -1e9;
			int best_pos = -1;
			board temp1(b), temp2(b);
			for (const int pos : space) {
				if (b(pos) != 0) continue;
				float value = 0;
				temp1(pos) = 1, temp2(pos) = 3;
				value += min_node(level, temp1) * 0.75;
				value += min_node(level, temp2) * 0.25;
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
			return action();
		}
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

	void switch_tuple_network(const int threshold) {
		const std::vector<int> available_network = {0, 5, 7, 10, 13, 15, 16, 17, 18, 19};
		if(std::find(available_network.begin(), available_network.end(), threshold) != available_network.end()) {
			std::string weight_filename = "ew_" + std::to_string(threshold);
			tn.load_weights(weight_filename);
		}
	}

	int get_search_level(const board& b) const {
		const int empty_tiles = b.get_empty_tile_count();
		if (empty_tiles < 2)
			return 5;
		else if (empty_tiles < 5)
			return 3;
		else
			return 1; 
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
	bool enable_search;
	int threshold;
	int prev_max_tile;
};
