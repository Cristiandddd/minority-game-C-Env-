#ifndef _MYEXCEPTIONS_H_
#define _MYEXCEPTIONS_H_

#include <exception>

class BadAlloc : public std::exception {
public:
    const char* what() const noexcept override {
        return "Memory allocation failed";
    }
};

class NotEqualSize : public std::exception {
public:
    const char* what() const noexcept override {
        return "Size mismatch error";
    }
};

class OutOfBounds : public std::exception {
public:
    const char* what() const noexcept override {
        return "Index out of bounds";
    }
};

class UnknownError : public std::exception {
public:
    const char* what() const noexcept override {
        return "Unknown error occurred";
    }
};

class Errors : public std::exception {
public:
    const char* what() const noexcept override {
        return "Generic error";
    }
};

#endif
