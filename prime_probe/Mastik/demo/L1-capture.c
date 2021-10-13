#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <util.h>
#include <l1.h>

#include <time.h>

#define MAX_SAMPLES 100000

void usage(const char *prog) {
  fprintf(stderr, "Usage: %s <output file>\n", prog);
  exit(1);
}

int main(int ac, char **av) {
  int samples = 1;

  if (av[1] == NULL)
    usage(av[0]);

  if (samples < 0)
    usage(av[0]);
  
  // int idx = atoi(av[1]);
  char *name = av[1];
  
  if (samples > MAX_SAMPLES)
    samples = MAX_SAMPLES;
  l1pp_t l1 = l1_prepare();

  int nsets = l1_getmonitoredset(l1, NULL, 0);

  // printf("nsets: %d\n", nsets);
  // printf("L1_SETS: %d\n", L1_SETS);

  int *map = calloc(nsets, sizeof(int));
  l1_getmonitoredset(l1, map, nsets);

  int rmap[L1_SETS];
  for (int i = 0; i < L1_SETS; i++)
    rmap[i] = -1;
  for (int i = 0; i < nsets; i++)
    rmap[map[i]] = i;
  

  uint16_t *res = calloc(samples * nsets, sizeof(uint16_t));
  for (int i = 0; i < samples * nsets; i+= 4096/sizeof(uint16_t))
    res[i] = 1;
  
  freopen(name, "w", stdout);
  
  for (;;) {
    l1_probe(l1, res);
    
    // printf("%lu ", (unsigned long)time(NULL)); 

    struct timespec current;
    clock_gettime(CLOCK_REALTIME, &current);

    uint64_t sec = (current.tv_sec);
    uint64_t nsec = (current.tv_nsec);
    printf("%lu.%lu ", sec, nsec);

    for (int i = 0; i < samples; i++) {
      for (int j = 0; j < L1_SETS; j++) {
        if (rmap[j] == -1) {
          // printf("  0 ");
          printf("0 ");
        }
        else {
          // printf("%d ", res[i*nsets + rmap[j]]);
          if (res[i*nsets + rmap[j]] < 230) { // AMD: 230, Intel: 120
            printf("1 ");
          }
          else {
            printf("0 ");
          }
        }
    
      }
      putchar('\n');
    }

    // delayloop(10000);
  }

  free(map);
  free(res);
  l1_release(l1);
}
