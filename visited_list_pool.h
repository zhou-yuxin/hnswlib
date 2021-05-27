#pragma once

#include <list>
#include <mutex>

#include <stdint.h>
#include <string.h>

namespace hnswlib {

typedef uint16_t vl_type;

class VisitedList {

private:
    vl_type tag;
    vl_type* array;
    size_t nelement;

public:
    VisitedList(size_t nelement) : nelement(nelement) {
        tag = vl_type(-1);
        array = new vl_type[nelement];
    }

    void reset() {
        tag++;
        if(tag == 0) {
            memset(array, 0, nelement * sizeof(vl_type));
            tag++;
        }
    };

    inline vl_type getTag() const {
        return tag;
    }

    inline vl_type* getArray() const {
        return array;
    }

    ~VisitedList() {
        delete[] array;
    }

};

class VisitedListPool {

private:
    size_t nelement;
    std::list<VisitedList*> pool;
    std::mutex lock;

public:
    VisitedListPool(size_t nelement, size_t ninstance = 1) : nelement(nelement) {
        for(size_t i = 0; i < ninstance; i++) {
            pool.push_back(new VisitedList(nelement));
        }
    }

    VisitedList* getFreeVisitedList() {
        VisitedList* instance;
        std::unique_lock<std::mutex> l(lock);
        if(pool.empty()) {
            instance = new VisitedList(nelement);
        }
        else {
            instance = pool.back();
            pool.pop_back();
        }
        l.unlock();
        instance->reset();
        return instance;
    }

    void releaseVisitedList(VisitedList* instance) {
        std::unique_lock<std::mutex> l(lock);
        pool.push_back(instance);
    }

    ~VisitedListPool() {
        while(!pool.empty()) {
            VisitedList* instance = pool.back();
            pool.pop_back();
            delete instance;
        }
    }

};

}