#include "stdio.h"

int main() {
    printf("%d %d", __STDC_VERSION__, __STDC_NO_VLA__);

    return 0;
}