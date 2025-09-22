#include "bed_handle.h"

void BedHandle::write_record(BedRecord *record) {
    // TODO: implement
    ;
}

bool BedHandle::has_next() { return ret_code_ >= 0; }   

bool BedHandle::next(BedRecord *record) {
    // TODO: implement
    return false;
}

void BedHandle::close() {
    // TODO: implement
    return;
}