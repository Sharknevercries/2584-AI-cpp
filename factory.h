#pragma once
#include "agent.h"
#include "player/dummy.h"
#include "player/tn.h"
#include "player/tn_mul.h"
#include "evil/random.h"
#include "evil/tn_p.h"
#include "evil/tn_p_mul.h"
#include "evil/tn.h"

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
        else if (type == "tn_mul")
            return new tn_mul_player("name=tn_mul " + args);
        else if (type == "maple")
            return new tn_mul_player("name=maple alpha=0 search=1 " + args);
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
            return new tn_p_evil("name=vanilla search=1 " + args);
        else if (type == "tn_p_mul")
            return new tn_p_mul_evil("name=tn_p_mul " + args);
        else if (type == "cinnamon")
            return new tn_p_mul_evil("name=cinnamon search=1 " + args);
        else if (type == "tn")
            return new tn_evil("name=tn " + args);
        else if (type == "coconut")
            return new tn_evil("name=coconut search=1 threshold=-1 alpha=0" + args);
        return new random_evil("name=random " + args);
    }
};