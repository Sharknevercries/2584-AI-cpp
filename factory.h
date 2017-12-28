#pragma once
#include "agent.h"
#include "player/dummy.h"
#include "player/tn.h"
#include "evil/random.h"
#include "evil/tn_p.h"

class player_factory {
public:
    static agent* create_player(std::string args = "") {
        size_t type_pos = args.find("type=");
        std::string type = "";
        if (type_pos != std::string::npos) {
            size_t space_pos = args.find(" ", type_pos);
            if (space_pos == std::string::npos)
                type = args.substr(type_pos + 5);
            else
                type = args.substr(type_pos + 5, space_pos - type_pos - 5);
        }

        if (type == "dummy")
            return new dummy_player("name=dummy " + args);
        else if (type == "tn")
            return new tn_player("name=tn " + args);
        else if (type == "chocola")            
            return new tn_player("name=chocola alpha=0 search=1 " + args);
        return new dummy_player("name=dummy " + args);
    }    
};

class evil_factory {
public:
    static agent* create_evil(std::string args) {
        size_t type_pos = args.find("type=");
        std::string type = "";
        if (type_pos != std::string::npos) {
            size_t space_pos = args.find(" ", type_pos);
            if (space_pos == std::string::npos)
                type = args.substr(type_pos + 5);
            else
                type = args.substr(type_pos + 5, space_pos - type_pos - 5);
        }

        if (type == "random")
            return new random_evil("name=random " + args);
        else if (type == "tn_p")
            return new tn_p_evil("name=tn_p " + args);
        else if (type == "vanilla")
            return new tn_p_evil("name=vanilla search=1" + args);
        return new random_evil("name=random " + args);
    }
};