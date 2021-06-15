#pragma once

#include <list>
#include <atomic>
#include <random>
#include <vector>
#include <cassert>
#include <unordered_map>
#include <unordered_set>

#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>

#include "hnswlib.h"
#include "visited_list_pool.h"

namespace hnswlib {

typedef uint32_t listsize_t;
typedef uint32_t id_t;
typedef uint8_t level_t;

template <typename Tdist>
class HierarchicalNSW : public AlgorithmInterface<Tdist> {

private:
    struct LinkList {
        listsize_t size : sizeof(listsize_t) * 8 - 2;
        listsize_t status : 2;
        id_t ids[];
    };

    enum Status {
        FREE = 0,
        ENABLED = 1,
        DISABLED = 2,
    };

    struct SpinLock {
        volatile uint8_t* byte;
        uint8_t mask;

        SpinLock(uint8_t* bitlocks, id_t id) {
            byte = bitlocks + id / 8;
            mask = 1 << (id % 8);
            while(__sync_fetch_and_or(byte, mask) & mask);
        }

        ~SpinLock() {
            uint8_t snap = __sync_fetch_and_and(byte, ~mask);
            assert(snap & mask);
        }
    };

    struct ReaderLock {
        pthread_rwlock_t* lock;

        inline ReaderLock(pthread_rwlock_t* lock): lock(lock) {
            int ret = pthread_rwlock_rdlock(lock);
            assert(ret == 0);
        }

        inline ~ReaderLock() {
            int ret = pthread_rwlock_unlock(lock);
            assert(ret == 0);
        }
    };

    struct WriterLock {
        pthread_rwlock_t* lock;

        inline WriterLock(pthread_rwlock_t* lock): lock(lock) {
            int ret = pthread_rwlock_wrlock(lock);
            assert(ret == 0);
        }

        inline ~WriterLock() {
            int ret = pthread_rwlock_unlock(lock);
            assert(ret == 0);
        }
    };

    struct CompareByFirst {
        inline bool operator()(const std::pair<Tdist, id_t>& a, const std::pair<Tdist, id_t>& b) const {
            return a.first < b.first;
        }
    };

    using candidates_t = std::priority_queue<std::pair<Tdist, id_t>, std::vector<std::pair<Tdist, id_t>>,
            CompareByFirst>;

private:
    size_t d_;
    size_t M_;
    size_t max_M_;
    size_t max_M0_;
    size_t ef_construction_;

    double mult_;
    size_t raw_data_size_;
    size_t data_size_;
    size_t size_links_level0_;
    size_t size_data_per_element_;
    size_t size_links_per_element_;
    size_t data_offset_;
    size_t label_offset_;

    VPFUNC vp_func_;
    DISTFUNC<Tdist> dist_func_;
    DISTFUNC<Tdist> st_dist_func_;

    int enterpoint_level_;
    id_t enterpoint_id_;

    size_t max_elements_;

    level_t* levels_;
    void** linklists_;
    uint8_t* bitlocks_;

    Level0StorageInterface* level0_storage_;
    void* level0_raw_memory_;

    std::vector<id_t> free_ids_;
    std::unordered_map<label_t, id_t> label_lookup_;

    std::default_random_engine random_;

    std::mutex global_lock_;
    pthread_rwlock_t add_rwlock_;

    size_t ef_;
    VisitedListPool* visited_list_pool_;

public:
    mutable std::atomic<uint64_t> metric_hops;
    mutable std::atomic<uint64_t> metric_distance_computations;

private:
    inline label_t* getExternalLabeLp(id_t id) const {
        return (label_t*)(size_t(level0_raw_memory_) + id * size_data_per_element_ + label_offset_);
    }

    inline label_t getExternalLabel(id_t id) const {
        return *getExternalLabeLp(id);
    }

    inline void setExternalLabel(id_t id, label_t label) {
        *getExternalLabeLp(id) = label;
    }

    inline void* getData(id_t id) const {
        return (void*)(size_t(level0_raw_memory_) + id * size_data_per_element_ + data_offset_);
    }

    inline LinkList* getLinkList0(id_t id) const {
        return (LinkList*)(size_t(level0_raw_memory_) + id * size_data_per_element_);
    };

    inline LinkList* getLinkListN(id_t id, int level) const {
        return (LinkList*)(size_t(linklists_[id]) + (level - 1) * size_links_per_element_);
    };

    inline LinkList* getLinkList(id_t id, int level) const {
        return level == 0 ? getLinkList0(id) : getLinkListN(id, level);
    };

    inline Status getStatus(id_t id) const {
        return getLinkList0(id)->status;
    }

    inline bool isFree(id_t id) const {
        return getLinkList0(id)->status == Status::FREE;
    }

    inline bool isEnabled(id_t id) const {
        return getLinkList0(id)->status == Status::ENABLED;
    }

    inline bool isDisabled(id_t id) const {
        return getLinkList0(id)->status == Status::DISABLED;
    }

    inline void setStatus(id_t id, Status status) {
        getLinkList0(id)->status = status;
    }

    inline int getRandomLevel() {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(random_)) * mult_;
        return (int)r;
    }

    candidates_t searchLayer(id_t ep_id, const void* data_point, int level) const {
        VisitedList* vl = visited_list_pool_->getFreeVisitedList();
        vl_type* visited_array = vl->getArray();
        vl_type visited_tag = vl->getTag();

        candidates_t top_candidates, candidate_set;
        Tdist lower_bound;

        if(isEnabled(ep_id)) {
            Tdist dist = dist_func_(data_point, getData(ep_id), d_);
            top_candidates.emplace(dist, ep_id);
            lower_bound = dist;
            candidate_set.emplace(-dist, ep_id);
        }
        else {
            lower_bound = std::numeric_limits<Tdist>::max();
            candidate_set.emplace(-lower_bound, ep_id);
        }
        visited_array[ep_id] = visited_tag;

        while(!candidate_set.empty()) {
            const std::pair<Tdist, id_t>& cur_pair = candidate_set.top();
            if((-cur_pair.first) > lower_bound) {
                break;
            }
            id_t cur_id = cur_pair.second;
            candidate_set.pop();
            
            SpinLock lock(bitlocks_, cur_id);

            const LinkList* linklist = getLinkList(cur_id, level);
#ifdef USE_SSE
            _mm_prefetch(visited_array + linklist->ids[0], _MM_HINT_T0);
            _mm_prefetch(getData(linklist->ids[0]), _MM_HINT_T0);
#endif
            for(listsize_t i = 0; i < linklist->size; i++) {
                id_t cand_id = linklist->ids[i];
#ifdef USE_SSE
                _mm_prefetch(visited_array + linklist->ids[i + 1], _MM_HINT_T0);
                _mm_prefetch(getData(linklist->ids[i + 1]), _MM_HINT_T0);
#endif
                if(visited_array[cand_id] == visited_tag) {
                    continue;
                }
                visited_array[cand_id] = visited_tag;
                Tdist dist = dist_func_(data_point, getData(cand_id), d_);
                if(top_candidates.size() < ef_construction_ || dist < lower_bound) {
                    if(isEnabled(cand_id)) {
                        top_candidates.emplace(dist, cand_id);
                        if(top_candidates.size() > ef_construction_) {
                            top_candidates.pop();
                        }
                        lower_bound = top_candidates.top().first;
                    }
                    candidate_set.emplace(-dist, cand_id);
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    template <bool collect_metrics>
    candidates_t searchBaseLayerST(id_t ep_id, const void* data_point, size_t ef) const {
        VisitedList* vl = visited_list_pool_->getFreeVisitedList();
        vl_type* visited_array = vl->getArray();
        vl_type visited_tag = vl->getTag();

        candidates_t top_candidates, candidate_set;
        Tdist lower_bound;

        if(isEnabled(ep_id)) {
            Tdist dist = st_dist_func_(data_point, getData(ep_id), d_);
            top_candidates.emplace(dist, ep_id);
            lower_bound = dist;
            candidate_set.emplace(-dist, ep_id);
        }
        else {
            lower_bound = std::numeric_limits<Tdist>::max();
            candidate_set.emplace(-lower_bound, ep_id);
        }
        visited_array[ep_id] = visited_tag;

        uint32_t hops = 0, distance_computations = 0;

        while(!candidate_set.empty()) {
            const std::pair<Tdist, id_t>& cur_pair = candidate_set.top();
            if((-cur_pair.first) > lower_bound) {
                break;
            }
            id_t cur_id = cur_pair.second;
            candidate_set.pop();

            const LinkList* linklist = getLinkList0(cur_id);
#ifdef USE_SSE
            _mm_prefetch(visited_array + linklist->ids[0], _MM_HINT_T0);
            _mm_prefetch(getData(linklist->ids[0]), _MM_HINT_T0);
#endif
            for(listsize_t i = 0; i < linklist->size; i++) {
                id_t cand_id = linklist->ids[i];
#ifdef USE_SSE
                _mm_prefetch(visited_array + linklist->ids[i + 1], _MM_HINT_T0);
                _mm_prefetch(getData(linklist->ids[i + 1]), _MM_HINT_T0);
#endif
                if(visited_array[cand_id] == visited_tag) {
                    continue;
                }
                visited_array[cand_id] = visited_tag;
                Tdist dist = st_dist_func_(data_point, getData(cand_id), d_);
                if(top_candidates.size() < ef || dist < lower_bound) {
                    if(isEnabled(cand_id)) {
                        top_candidates.emplace(dist, cand_id);
                        if(top_candidates.size() > ef) {
                            top_candidates.pop();
                        }
                        lower_bound = top_candidates.top().first;
                    }
                    candidate_set.emplace(-dist, cand_id);
                }
            }
            if(collect_metrics) {
                hops++;
                distance_computations += linklist->size;
            }
        }
        if(collect_metrics) {
            metric_hops += hops;
            metric_distance_computations += distance_computations;
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    std::vector<id_t> getNeighborsByHeuristic(candidates_t& top_candidates, size_t M) const {
        std::vector<id_t> selected;

        size_t ncand = top_candidates.size();
        if(ncand < M) {
            selected.resize(ncand);
            for(size_t i = ncand; i > 0; i--) {
                selected[i - 1] = top_candidates.top().second;
                top_candidates.pop();
            }
            return selected;
        }

        std::vector<std::pair<Tdist, id_t>> queue;
        queue.resize(ncand);
        for(size_t i = 0; i < ncand; i++) {
            queue[i] = top_candidates.top();
            top_candidates.pop();
        }
        
        selected.reserve(M);
        while(!queue.empty()) {
            if(selected.size() >= M) {
                break;
            }
            const std::pair<Tdist, id_t>& cur_pair = queue.back();
            Tdist cur_dist = cur_pair.first;
            id_t cur_id = cur_pair.second;
            queue.pop_back();

            bool good = true;
            for(id_t id : selected) {
                Tdist dist = dist_func_(getData(cur_id), getData(id), d_);
                if(dist < cur_dist) {
                    good = false;
                    break;
                }
            }
            if(good) {
                selected.push_back(cur_id);
            }
        }
        return selected;
    }

    id_t connectNeighbors(id_t id, candidates_t& top_candidates, int level, bool with_backward) {
        assert(!top_candidates.empty());

        size_t cur_max_M = level ? max_M_ : max_M0_;
        std::vector<id_t> selected = getNeighborsByHeuristic(top_candidates, cur_max_M);
        assert(!selected.empty() && selected.size() <= cur_max_M);

        {
            SpinLock lock(bitlocks_, id);
            LinkList* linklist = getLinkList(id, level);
            linklist->size = 0;
            memcpy(linklist->ids, selected.data(), selected.size() * sizeof(id_t));
            linklist->size = selected.size();
        }

        if(with_backward) {
            for(id_t nn_id : selected) {
                assert(nn_id != id);
                SpinLock lock(bitlocks_, nn_id);

                assert(levels_[nn_id] >= level);
                LinkList* linklist = getLinkList(nn_id, level);

                assert(linklist->size <= cur_max_M);
                if(linklist->size < cur_max_M) {
                    linklist->ids[linklist->size] = id;
                    linklist->size++;
                }
                else {
                    const void* data = getData(nn_id);
                    candidates_t candidates;
                    candidates.emplace(dist_func_(data, getData(id), d_), id);
                    for(listsize_t i = 0; i < linklist->size; i++) {
                        id_t id_i = linklist->ids[i];
                        candidates.emplace(dist_func_(data, getData(id_i), d_), id_i);
                    }
                    std::vector<id_t> nn_selected = getNeighborsByHeuristic(candidates, cur_max_M);
                    linklist->size = 0;
                    memcpy(linklist->ids, nn_selected.data(), nn_selected.size() * sizeof(id_t));
                    linklist->size = nn_selected.size();
                }
            }
        }

        return selected.front();
    }

public:
    HierarchicalNSW(SpaceInterface<Tdist>* space, Level0StorageInterface* level0_storage, size_t max_elements,
            size_t M = 16, size_t ef_construction = 200, int random_seed = 100) {
        d_ = space->getDim();
        M_ = M;
        max_M_ = M_;
        max_M0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);

        mult_ = 1.0 / log(1.0 * M_);
        raw_data_size_ = space->getRawDataSize();
        data_size_ = space->getDataSize();
        size_links_level0_ = max_M0_ * sizeof(id_t) + sizeof(listsize_t);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(label_t);
        size_links_per_element_ = max_M_ * sizeof(id_t) + sizeof(listsize_t);
        data_offset_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;

        vp_func_ = space->getVectorPreprocessFunc();
        dist_func_ = space->getDistFunc();
        st_dist_func_ = space->getDistFuncST();

        enterpoint_level_ = -1;
        enterpoint_id_ = id_t(-1);

        max_elements_ = max_elements;

        levels_ = (level_t*)malloc(sizeof(level_t) * max_elements_);
        if(levels_ == nullptr) {
            throw std::runtime_error("out of memory"); 
        }
        memset(levels_, 0, sizeof(level_t) * max_elements_);

        linklists_ = (void**)malloc(sizeof(void*) * max_elements_);
        if(linklists_ == nullptr) {
            throw std::runtime_error("out of memory");
        }

        size_t bitlock_len = max_elements_ / 8 + 1;
        bitlocks_ = (uint8_t*)malloc(bitlock_len);
        if(bitlocks_ == nullptr) {
            throw std::runtime_error("out of memory"); 
        }
        memset(bitlocks_, 0, bitlock_len);

        level0_storage_ = level0_storage;
        size_t level0_len = size_data_per_element_ * max_elements_;
        level0_raw_memory_ = level0_storage_->allocate(level0_len);
        memset(level0_raw_memory_, 0, level0_len);

        for(size_t i = 0; i < max_elements_; i++) {
            free_ids_.push_back(i);
        }

        random_.seed(random_seed);

        int ret = pthread_rwlock_init(&add_rwlock_, nullptr);
        assert(ret == 0);

        ef_ = 10;
        visited_list_pool_ = new VisitedListPool(max_elements_);

        metric_hops.store(0);
        metric_distance_computations.store(0);

        delete space;
    }

    ~HierarchicalNSW() {
        for(id_t id = 0; id < max_elements_; id++) {
            if(levels_[id] > 0) {
                free(linklists_[id]);
            }
        }
        free(levels_);
        free(bitlocks_);
        free(linklists_);
        level0_storage_->free(level0_raw_memory_, max_elements_ * size_data_per_element_);
        delete level0_storage_;
        delete visited_list_pool_;
        int ret = pthread_rwlock_destroy(&add_rwlock_);
        assert(ret == 0);
    }

    inline size_t getM() const {
        return M_;
    }

    inline size_t getEfConstruction() const {
        return ef_construction_;
    }

    inline void setEfConstruction(size_t ef) {
        ef_construction_ = ef;
    }

    inline size_t getEfSearch() const {
        return ef_;
    }

    inline void setEfSearch(size_t ef) {
        ef_ = ef;
    }

    inline size_t getMaxCount() const {
        return max_elements_;
    }

    inline size_t getFreeCount() const {
        return free_ids_.size();
    }

    inline void addPoint(const void* data_point, label_t label) override {
        addPoint(data_point, label, -1);
    }

    id_t addPoint(const void* data_point, label_t label, int level) {
        ReaderLock add_rlock(&add_rwlock_);
        std::unique_lock<std::mutex> glock(global_lock_);
        auto found = label_lookup_.find(label);
        if(found != label_lookup_.end()) {
            throw std::runtime_error("this label is already used");
        }
        if(free_ids_.empty()) {
            throw std::runtime_error("the number of elements exceeds the limit");
        }
        id_t id = free_ids_.back();
        free_ids_.pop_back();
        label_lookup_[label] = id;
        if(level < 0) {
            level = getRandomLevel();
        }
        int ep_level = enterpoint_level_;
        id_t cur_ep_id = enterpoint_id_;
        if(level <= ep_level) {
            glock.unlock();
        }

        void* data = getData(id);
        {
            levels_[id] = level;
            LinkList* linklist = getLinkList0(id);
            linklist->size = 0;
            linklist->status = Status::ENABLED;
            memcpy(data, data_point, raw_data_size_);
            if(vp_func_) {
                vp_func_(data, d_);
            }
            setExternalLabel(id, label);
            if(level) {
                size_t listsize = size_links_per_element_ * level;
                void* linklist = malloc(listsize);
                if(linklist == nullptr) {
                    throw std::runtime_error("out of memory");
                }
                memset(linklist, 0, listsize);
                linklists_[id] = linklist;
            }
        }

        if(cur_ep_id != id_t(-1)) {
            if(level < ep_level) {
                Tdist min_dist = dist_func_(data, getData(cur_ep_id), d_);
                for(int l = ep_level; l > level; l--) {
                    bool changed = true;
                    while(changed) {
                        changed = false;
                        SpinLock lock(bitlocks_, cur_ep_id);
                        const LinkList* linklist = getLinkListN(cur_ep_id, l);
                        for(listsize_t i = 0; i < linklist->size; i++) {
                            id_t id_i = linklist->ids[i];
                            Tdist dist_i = dist_func_(data, getData(id_i), d_);
                            if(dist_i < min_dist) {
                                min_dist = dist_i;
                                cur_ep_id = id_i;
                                changed = true;
                            }
                        }
                    }
                }
            }

            for(int l = std::min(level, ep_level); l >= 0; l--) {
                candidates_t top_candidates = searchLayer(cur_ep_id, data, l);
                if(!top_candidates.empty()) {
                    cur_ep_id = connectNeighbors(id, top_candidates, l, true);
                }
            }
        }

        if(level > ep_level) {
            enterpoint_level_ = level;
            enterpoint_id_ = id;
        }

        return id;
    }

    void disablePoint(label_t label) {
        std::unique_lock<std::mutex> glock(global_lock_);
        auto found = label_lookup_.find(label);
        if(found == label_lookup_.end()) {
            throw std::runtime_error("label not exists");
        }
        id_t id = found->second;
        label_lookup_.erase(found);
        setStatus(id, Status::DISABLED);
    }

    void recycleDisabledPoints(size_t nthread = 1) {
        std::unique_lock<std::mutex> glock(global_lock_);
        if(isDisabled(enterpoint_id_)) {
            bool found = false;
            {
                SpinLock lock(bitlocks_, enterpoint_id_);
                const LinkList* linklist = getLinkList(enterpoint_id_, enterpoint_level_);
                for(listsize_t i = 0; i < linklist->size; i++) {
                    id_t id_i = linklist->ids[i];
                    if(isEnabled(id_i)) {
                        enterpoint_id_ = id_i;
                        found = true;
                        break;
                    }
                }
            }

            if(!found) {
                int alt_level = -1;
                id_t alt_id = id_t(-1);
                #pragma omp parallel num_threads(nthread)
                {
                    int alt_level_local = -1;
                    id_t alt_id_local = id_t(-1);
                    #pragma omp for
                    for(id_t id = 0; id < max_elements_; id++) {
                        if(isEnabled(id)) {
                            if(int(levels_[id]) > alt_level_local) {
                                alt_level_local = levels_[id];
                                alt_id_local = id;
                            }
                        }
                    }
                    #pragma omp critical
                    if(alt_level_local > alt_level) {
                        alt_level = alt_level_local;
                        alt_id = alt_id_local;
                    }
                }

                if(alt_id == id_t(-1)) {
                    throw std::runtime_error("enterpoint is the only enabled point");
                }
                enterpoint_level_ = alt_level;
                enterpoint_id_ = alt_id;
            }
        }
        assert(isEnabled(enterpoint_id_));
        glock.unlock();

        std::vector<id_t> disabled_ids;

        #pragma omp parallel for num_threads(nthread) schedule(dynamic)
        for(id_t id = 0; id < max_elements_; id++) {
            if(isFree(id)) {
                continue;
            }
            if(isDisabled(id)) {
                #pragma omp critical
                disabled_ids.push_back(id);
            }

            for(int l = levels_[id]; l >= 0; l--) {
                std::vector<id_t> enabled_ids, disabled_ids;
                enabled_ids.reserve(l ? max_M_ : max_M0_);

                {
                    SpinLock lock(bitlocks_, id);
                    const LinkList* linklist = getLinkList(id, l);
                    for(listsize_t i = 0; i < linklist->size; i++) {
                        id_t id_i = linklist->ids[i];
                        assert(id_i != id);
                        if(isEnabled(id_i)) {
                            enabled_ids.push_back(id_i);
                        }
                        else {
                            assert(isDisabled(id_i));
                            disabled_ids.push_back(id_i);
                        }
                    }
                }
                if(disabled_ids.empty()) {
                    continue;
                }

                candidates_t candidates;
                std::unordered_set<id_t> mask_set;
                mask_set.insert(id);
                const void* data = getData(id);

                for(id_t id_i : enabled_ids) {
                    if(mask_set.insert(id_i).second) {
                        candidates.emplace(dist_func_(data, getData(id_i), d_), id_i);
                    }
                }
                for(id_t id_i : disabled_ids) {
                    SpinLock lock(bitlocks_, id_i);
                    const LinkList* linklist = getLinkList(id_i, l);
                    for(listsize_t j = 0; j < linklist->size; j++) {
                        id_t id_j = linklist->ids[j];
                        assert(id_j != id_i);
                        if(isEnabled(id_j)) {
                            if(mask_set.insert(id_j).second) {
                                candidates.emplace(dist_func_(data, getData(id_j), d_), id_j);
                            }
                        }
                    }
                }

                if(candidates.empty()) {
                    SpinLock lock(bitlocks_, id);
                    LinkList* linklist = getLinkList(id, l);
                    linklist->size = 0;
                }
                else {
                    connectNeighbors(id, candidates, l, isEnabled(id));
                }
            }
        }

        {
            WriterLock add_wlock(&add_rwlock_);
        }

        size_t bitmap_len = max_elements_ / 8 + 1;
        uint8_t* linked_bitmap = new uint8_t [bitmap_len];
        memset(linked_bitmap, 0, bitmap_len);

        #pragma omp parallel for num_threads(nthread) schedule(dynamic)
        for(id_t id = 0; id < max_elements_; id++) {
            if(isFree(id)) {
                continue;
            }
            for(int l = levels_[id]; l >= 0; l--) {
                SpinLock lock(bitlocks_, id);
                const LinkList* linklist = getLinkList(id, l);
                for(listsize_t i = 0; i < linklist->size; i++) {
                    id_t id_i = linklist->ids[i];
                    __sync_fetch_and_or(linked_bitmap + id_i / 8, 1 << (id_i % 8));
                }
            }
        }

        #pragma omp parallel for num_threads(nthread)
        for(size_t i = 0; i < disabled_ids.size(); i++) {
            id_t id = disabled_ids[i];
            if(!(linked_bitmap[id / 8] & (1 << (id % 8)))) {
                setStatus(id, Status::FREE);
                if(levels_[id] > 0) {
                    free(linklists_[id]);
                    levels_[id] = 0;
                }
                #pragma omp critical
                free_ids_.push_back(id);
            }
        }

        delete linked_bitmap;
    }

    inline std::priority_queue<std::pair<Tdist, label_t>> searchKnn(const void* data_point, size_t k)
            const override {
        return searchKnn<false>(k, data_point);
    }

    template <bool collect_metrics>
    std::priority_queue<std::pair<Tdist, label_t>> searchKnn(size_t k, const void* data_point) const {
        std::priority_queue<std::pair<Tdist, label_t>> result;
        if(getMaxCount() == getFreeCount()) {
            return result;
        }

        id_t ep_id = enterpoint_id_;
        Tdist cur_dist = st_dist_func_(data_point, getData(ep_id), d_);

        uint64_t hops = 0, distance_computations = 0;
        for(int l = enterpoint_level_; l > 0; l--) {
            bool changed = true;
            while(changed) {
                changed = false;
                const LinkList* linklist = getLinkListN(ep_id, l);
#ifdef USE_SSE
                _mm_prefetch(getData(linklist->ids[0]), _MM_HINT_T0);
#endif
                for(listsize_t i = 0; i < linklist->size; i++) {
                    id_t id_i = linklist->ids[i];
#ifdef USE_SSE
                    _mm_prefetch(getData(linklist->ids[i + 1]), _MM_HINT_T0);
#endif
                    Tdist dist_i = st_dist_func_(data_point, getData(id_i), d_);
                    if(dist_i < cur_dist) {
                        cur_dist = dist_i;
                        ep_id = id_i;
                        changed = true;
                    }
                }
                if(collect_metrics) {
                    hops++;
                    distance_computations += linklist->size;
                }
            }
        }
        if(collect_metrics) {
            metric_hops += hops;
            metric_distance_computations += distance_computations;
        }

        candidates_t top_candidates = searchBaseLayerST<collect_metrics>(ep_id, data_point, std::max(ef_, k));
 
        while(top_candidates.size() > k) {
            top_candidates.pop();
        }
        while(top_candidates.size() > 0) {
            const std::pair<Tdist, id_t>& pair = top_candidates.top();
            result.push(std::pair<Tdist, label_t>(pair.first, getExternalLabel(pair.second)));
            top_candidates.pop();
        }
        return result;
    }

private:
    static void write(FILE* file, const void* buf, size_t len) {
        if(fwrite(buf, 1, len, file) != len) {
            throw std::runtime_error("failed to write to file");
        }
    }

    template <typename T>
    static void write(FILE* file, const T &pod) {
        write(file, &pod, sizeof(T));
    }

    static void read(FILE* file, void* buf, size_t len) {
        if(fread(buf, 1, len, file) != len) {
            throw std::runtime_error("failed to read from file");
        }
    }

    template <typename T>
    static void read(FILE* file, T& pod) {
        read(file, &pod, sizeof(T));
    }

    static size_t alignToPageSize(size_t offset) {
        size_t pgsize = getpagesize();
        size_t align_mod = offset % pgsize;
        return align_mod == 0 ? offset : offset + pgsize - align_mod;
    }

    HierarchicalNSW() {}

public:
    void save(FILE* file) const {
        write(file, M_);
        write(file, max_M_);
        write(file, max_M0_);
        write(file, ef_construction_);
        write(file, ef_);

        write(file, enterpoint_level_);
        write(file, enterpoint_id_);

        write(file, max_elements_);

        for(id_t id = 0; id < max_elements_; id++) {
            int level = levels_[id];
            write(file, level);
            if(level) {
                write(file, linklists_[id], size_links_per_element_ * level);
            }
        }

        auto padding_to_pgsize = [&]() {
            long position = ftell(file);
            if(position < 0) {
                throw std::runtime_error("failed to get the current position of file");
            }
            size_t pgsize = getpagesize();
            size_t align_mod = position % pgsize;
            if(align_mod != 0) {
                size_t padding_len = pgsize - align_mod;
                char* padding = new char [padding_len];
                write(file, padding, padding_len);
                delete[] padding;
            }
        };
        padding_to_pgsize();
        write(file, level0_raw_memory_, max_elements_ * size_data_per_element_);
        padding_to_pgsize();
    }

    static HierarchicalNSW<Tdist>* load(FILE* file, SpaceInterface<Tdist>* space,
            Level0StorageInterface* level0_storage) {
        HierarchicalNSW<Tdist>* hnsw = new HierarchicalNSW<Tdist>;

        hnsw->d_ = space->getDim();

        read(file, hnsw->M_);
        read(file, hnsw->max_M_);
        read(file, hnsw->max_M0_);
        read(file, hnsw->ef_construction_);
        read(file, hnsw->ef_);

        hnsw->mult_ = 1.0 / log(1.0 * hnsw->M_);
        hnsw->raw_data_size_ = space->getRawDataSize();
        hnsw->data_size_ = space->getDataSize();
        hnsw->size_links_level0_ = hnsw->max_M0_ * sizeof(id_t) + sizeof(listsize_t);
        hnsw->size_data_per_element_ = hnsw->size_links_level0_ + hnsw->data_size_ + sizeof(label_t);
        hnsw->size_links_per_element_ = hnsw->max_M_ * sizeof(id_t) + sizeof(listsize_t);
        hnsw->data_offset_ = hnsw->size_links_level0_;
        hnsw->label_offset_ = hnsw->size_links_level0_ + hnsw->data_size_;

        hnsw->vp_func_ = space->getVectorPreprocessFunc();
        hnsw->dist_func_ = space->getDistFunc();
        hnsw->st_dist_func_ = space->getDistFuncST();

        read(file, hnsw->enterpoint_level_);
        read(file, hnsw->enterpoint_id_);

        read(file, hnsw->max_elements_);

        hnsw->levels_ = (level_t*)malloc(sizeof(level_t) * hnsw->max_elements_);
        if(hnsw->levels_ == nullptr) {
            throw std::runtime_error("out of memory"); 
        }
        memset(hnsw->levels_, 0, sizeof(level_t) * hnsw->max_elements_);

        hnsw->linklists_ = (void**)malloc(sizeof(void*) * hnsw->max_elements_);
        if(hnsw->linklists_ == nullptr) {
            throw std::runtime_error("out of memory");
        }

        size_t bitlock_len = hnsw->max_elements_ / 8 + 1;
        hnsw->bitlocks_ = (uint8_t*)malloc(bitlock_len);
        if(hnsw->bitlocks_ == nullptr) {
            throw std::runtime_error("out of memory");
        }
        memset(hnsw->bitlocks_, 0, bitlock_len);

        for(id_t id = 0; id < hnsw->max_elements_; id++) {
            int level;
            read(file, level);
            hnsw->levels_[id] = level;
            if(level) {
                size_t len = hnsw->size_links_per_element_ * level;
                void* linklist = malloc(len);
                if(linklist == nullptr) {
                    throw std::runtime_error("out of memory"); 
                }
                read(file, linklist, len);
                hnsw->linklists_[id] = linklist;
            }
        }

        hnsw->level0_storage_ = level0_storage;
        long position = ftell(file);
        if(position < 0) {
            throw std::runtime_error("failed to get the current position of file");
        }
        size_t level0_offset = alignToPageSize(position);
        size_t level0_len = hnsw->size_data_per_element_ * hnsw->max_elements_;
        hnsw->level0_raw_memory_ = level0_storage->load(file, level0_offset, level0_len);
        if(fseek(file, alignToPageSize(level0_offset + level0_len), SEEK_SET) < 0) {
            throw std::runtime_error("failed to set current position of file");
        }

        for(id_t id = 0; id < hnsw->max_elements_; id++) {
            if(hnsw->isFree(id)) {
                hnsw->free_ids_.push_back(id);
            }
            else if(hnsw->isEnabled(id)) {
                hnsw->label_lookup_[hnsw->getExternalLabel(id)] = id;
            }
        }

        hnsw->random_.seed(time(nullptr));

        int ret = pthread_rwlock_init(&(hnsw->add_rwlock_), nullptr);
        assert(ret == 0);

        hnsw->visited_list_pool_ = new VisitedListPool(hnsw->max_elements_);

        hnsw->metric_hops.store(0);
        hnsw->metric_distance_computations.store(0);
        
        delete space;

        return hnsw;
    }

};

}
