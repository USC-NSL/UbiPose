
#include <absl/log/log.h>

int main(int argc, char *argv[]) {

  LOG(FATAL) << "Fatal message";
  return 0;
}
