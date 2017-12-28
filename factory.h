#pragma once
#include "agent.h"

class player_factory {
public:
    static agent create_player(std::string args = "") {
        size_t type_pos = args.find("type=");
        std::string type;
        if (type_pos == std::string::npos)
            type = "dummy";
        else {
            size_t space_pos = args.find(" ", type_pos);
            type = args.substr(type_pos + 5, space_pos - type_pos - 5);
        }

        if (type == "dummy")
            // TODO
            ;
        else if (type == "tn")
            // TODO
            ;
        else if (type == "chocola")
            // TODO
            ;
    }    
};

class evil_factory {
public:
    static agent create_evil(std::string args) {
        size_t type_pos = args.find("type=");
        std::string type;
        if (type_pos == std::string::npos)
            type = "random";
        else {
            size_t space_pos = args.find(" ", type_pos);
            type = args.substr(type_pos + 5, space_pos - type_pos - 5);
        }

        if (type == "random")
            // TODO
            ;
        else if (type == "vanilla")
            // TODO
            ;
    }
};