#pragma once
#include "../agent.h"
#include "../tuple_network.h"

class tn_p_mul_evil : public agent {
public:
	tn_p_mul_evil(const std::string& args = "") : agent("role=environment " + args), enable_search(false) {
		if (property.find("seed") != property.end())
			engine.seed(int(property["seed"]));
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

	virtual action take_action(const board& b) {
		std::uniform_int_distribution<int> popup(0, 3);
		bool given_tile = false;
		int tile = 0;
		if (property.find("tile") != property.end()) {
			tile = (int)property["tile"];
			given_tile = true;
		}
		else
			tile = popup(engine) ? 1 : 3;

		float m = 1e9, m4 = m * 4;
		int best_pos = -1;
		int level = 1;

		if (enable_search)
			level = get_search_level(b.get_empty_tile_count());
		
		board temp1(b), temp2(b);
		for (const int pos : space) {
			if (b(pos) != 0) continue;
			float v1, v2, ev;
			if (given_tile) {
				temp1(pos) = tile;
				ev = max_node(level, temp1, -1e9, m);
				temp1(pos) = 0;
			}
			else {
				temp1(pos) = 1, temp2(pos) = 3;
				v1 = max_node(level, temp1, -1e9, 1e9);
				v2 = max_node(level, temp2, -1e9, m4 - 3 * v1);
				ev = v1 * 0.75 + v2 * 0.25;
				temp1(pos) = 0, temp2(pos) = 0;
			}

			if (m > ev) {
				m = ev;
				m4 = m * 4;
				best_pos = pos;
			}
		}
		return best_pos == -1 ? action() : action::place(tile, best_pos);
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

	int get_search_level(const int empty_tiles) const {
		if (empty_tiles < 2)
			return 5;
		else if (empty_tiles < 5)
			return 3;
		else
			return 1; 
	}

	tuple_network& get_tn(const int empty_tiles) {
        return tns[empty_tiles / 2];
    }

private:
	std::default_random_engine engine;
	std::array<tuple_network, 8> tns;
	bool enable_search;
};
