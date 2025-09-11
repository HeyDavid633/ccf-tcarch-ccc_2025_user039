HIPCC = hipcc

TARGET = apsp

SRCS = main.cpp kernel.hip

HEADERS = main.h

CXXFLAGS = -O2 -w

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(HIPCC) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET) *.o