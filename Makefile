PZSDK_PATH?=/opt/pzsdk.ver4.1
DEFAULT_MAKE=$(PZSDK_PATH)/make/default_pzcl_host.mk

TARGET=pzcAdd
CPPSRC=main.cpp
CCOPT=-O2 -Wall -D__LINUX__ -DNDEBUG -std=c++11

INC_DIR?=

LIB_DIR?=

PZCL_KERNEL_DIRS=kernel

# supported archtecture:
# sc1-64, sc2
PZC_TARGET_ARCH?=sc2
export PZC_TARGET_ARCH

include $(DEFAULT_MAKE)

run:
	@./$(TARGET) 102400
