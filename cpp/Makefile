CXX := clang++
CXXFLAGS := -std=c++20 -O3 $(shell pkg-config --cflags libpcre2-8)
LDLIBS := $(shell pkg-config --libs libpcre2-8)

SOURCES := tokenizer.cpp main.cpp
TARGET := ../artifacts/tokenizer/cpp/tokenizer

$(TARGET): $(SOURCES) tokenizer.hpp
	mkdir -p $(dir $(TARGET))
	$(CXX) $(CXXFLAGS) $(SOURCES) $(LDLIBS) -o $(TARGET)

run: $(TARGET)
	$(TARGET)

extension:
	cd .. && .venv/bin/python -m pip install -e .

clean:
	rm -f $(TARGET)

.PHONY: run extension clean
