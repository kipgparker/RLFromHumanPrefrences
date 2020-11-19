class PrefBuffer:
    def __init__(self, garner, db_train, db_val, maxlen=1000, maxqlen=3):

        self.garner = garner
        self.train_db = db_train
        self.val_db = db_val
        self.val_fraction = self.val_db.maxlen / (self.val_db.maxlen +
                                                  self.train_db.maxlen)

        self.segments = CompressedDict()
        self.seg_refs = {}

        self.tested_pairs = set()
        self.queue = {}
        self.maxlen = maxlen
        self.maxqlen = maxqlen
        self.lock = Lock()
        self.stop_run = False

        #returned = garner.get_backlog(limit=maxlen)
        # for uuid, val in returned.items():
        # self.lock.acquire(blocking=True)
        # if np.random.rand() < self.val_fraction:
        #    self.val_db.append(val[0][0][:,:,:,0:3], val[0][1][:,:,:,0:3], 1 if val[1] else 0)
        # else:
        #    self.train_db.append(val[0][0][:,:,:,0:3], val[0][1][:,:,:,0:3],  1 if val[1] else 0)
        # self.lock.release()
        # Aquiring old currently not a good idea :/

    def select_prefs(self):
        segment_idxs = list(range(len(self.segments)))
        shuffle(segment_idxs)
        possible_pairs = combinations(segment_idxs, 2)

        keys = list(pref_buffer.segments.keys())

        for i1, i2 in possible_pairs:
            s1, s2 = self.segments[keys[i1]], self.segments[keys[i2]]
            if ((s1.hash, s2.hash) not in self.tested_pairs) and \
               ((s2.hash, s1.hash) not in self.tested_pairs):
                self.tested_pairs.add((s1.hash, s2.hash))
                self.tested_pairs.add((s2.hash, s1.hash))
                return s1.hash, s2.hash

    def recv_prefs(self):
        result = self.garner.query()

        print(result)

        if len(result) != 0:
            for key, value in result.items():
                k1, k2 = self.queue[key]

                if np.random.rand() < self.val_fraction:
                    self.val_db.append(
                        self.segments[k1].frames, self.segments[k2].frames, 1 if value else 0)
                else:
                    self.train_db.append(
                        self.segments[k1].frames, self.segments[k2].frames, 1 if value else 0)
            del self.queue[key]

    def put_prefs(self):
        try:
            while len(self.queue) <= self.maxqlen:
                print('put')
                k1, k2 = self.select_prefs()
                pref_id = self.garner.put(
                    [np.array(self.segments[k1].frames), np.array(self.segments[k1].frames)], False)
                self.queue[pref_id] = (k1, k2)
        except:
            print('No prefs to compare')

    def run(self):
        while not self.stop_run:
            time.sleep(1)
            self.put_prefs()
            self.recv_prefs()

    def start_thread(self):
        self.stop_run = False
        self.garner.connect()
        Thread(target=self.run).start()
        # self.run()

    def stop_thread(self):
        self.garner.disconnect()
        self.stop_recv = True

    def get_dbs(self):
        self.lock.acquire(blocking=True)
        train_copy = copy.deepcopy(self.train_db)
        val_copy = copy.deepcopy(self.val_db)
        self.lock.release()
        return train_copy, val_copy

    def add_segment(self, segment):

        k = segment.hash

        if k not in self.segments.keys():
            self.segments[k] = segment
            self.seg_refs[k] = 1
        else:
            self.seg_refs[k] += 1

        if len(self.prefs) > self.maxlen:
            self.del_first()

        #            k1 = hash(np.array(s1).tobytes())
        #k2 = hash(np.array(s2).tobytes())

    def del_first(self):
        self.del_pref(0)

    def del_pref(self, n):
        if n >= len(self.prefs):
            raise IndexError("Preference {} doesn't exist".format(n))
        k1, k2, _ = self.prefs[n]
        for k in [k1, k2]:
            if self.seg_refs[k] == 1:
                del self.segments[k]
                del self.seg_refs[k]
            else:
                self.seg_refs[k] -= 1
        del self.prefs[n]

    def __len__(self):
        return len(self.prefs)
