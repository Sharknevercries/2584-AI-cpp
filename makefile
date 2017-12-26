all:
	g++ -std=c++17 -O3 -g -Wall -fmessage-length=0 -o 2584 2584.cpp 
clean:
	rm 2584
test-play:
	./2584 --evil=evil=0
test-evil:
	./2584 --play=player=0