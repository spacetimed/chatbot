CXX := clang++
CXXFLAGS := -std=c++20 $(shell pkg-config --cflags libpcre2-8)
LDLIBS := $(shell pkg-config --libs libpcre2-8)

SOURCE := tokenizer.cpp
TARGET := ../artifacts/tokenizer/cpp/tokenizer

$(TARGET): $(SOURCE)
	mkdir -p $(dir $(TARGET))
	$(CXX) $(CXXFLAGS) $(SOURCE) $(LDLIBS) -o $(TARGET)

run: $(TARGET)
	$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: run clean
