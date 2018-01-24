#pragma once
#include "../agent.h"
#include "../tuple_network.h"

class tn_mul_player : public agent {
public:
	tn_mul_player(const std::string& args = "") : agent("role=player " + args), alpha(0.0025f), enable_search(false) {
		episode.reserve(32768);
		if (property.find("alpha") != property.end())
			alpha = float(property["alpha"]);
		if (property.find("search") != property.end())
			enable_search = ((int)property["search"] == 1) ? true : false;

        for (auto &tn : tns) {
            tn = tuple_network({
                {0, 1, 2, 3, 4},
                {0, 1, 4, 5, 6},
                {1, 2, 3, 5, 6},
                {5, 6, 7, 9, 10},
            });
        }

		if (property.find("load") != property.end()) {
            const std::string prefix = std::string("weights/") + static_cast<std::string>(property["load"]);
            int iter = 0;
            for (auto &tn : tns) {
                const std::string path = prefix + std::to_string(iter);
			    tn.load_weights(path);
                ++iter;
            }
        }
	}
	~tn_mul_player() {
		if (property.find("save") != property.end()) {
            const std::string prefix = std::string("weights/") + static_cast<std::string>(property["save"]);
            int iter = 0;
            for (auto &tn : tns) {
                const std::string path = prefix + std::to_string(iter);
			    tn.save_weights(path);
                ++iter;
            }
        }
	}

	virtual void open_episode(const std::string& flag = "") {
		episode.clear();
		episode.reserve(32768);
	}

	virtual void close_episode(const std::string& flag = "") {
		auto cur = episode.end() - 1, prev = cur - 1;
        auto& cur_tn = get_tn(cur->after.get_empty_tile_count());
		cur_tn.update(cur->after, alpha * (-cur_tn.estimate(cur->after)));		
		while (cur != episode.begin()) {
            auto& prev_tn = get_tn(prev->after.get_empty_tile_count());
            auto& cur_tn = get_tn(cur->after.get_empty_tile_count());
			float td_error = cur->reward + cur_tn.estimate(cur->after) - prev_tn.estimate(prev->after);
			prev_tn.update(prev->after, alpha * td_error);
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
			const int reward = act.apply(temp);
		    const int empty_tiles = temp.get_empty_tile_count();
			int level = 0;
			
            if (enable_search)
				level = get_search_level(empty_tiles);

			if (reward != -1) {
				float esti = 0;
				if (level > 0)
					esti = min_node(level, temp, best_value - reward, 1e9);
				else {
                    auto& tn = get_tn(empty_tiles);
					esti = tn.estimate(temp);
                }
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
				else {
                    auto& tn = get_tn(temp.get_empty_tile_count());
					esti = tn.estimate(temp);
                }
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

	int get_search_level(const int empty_tiles) const {
		if (empty_tiles < 3)
			return 8;
		else if (empty_tiles < 7)
			return 6;
		else
			return 4;
	}

    tuple_network& get_tn(const int empty_tiles) {
		return tns[empty_tiles / 2];
    }

private:
    std::array<tuple_network, 8> tns;

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
};