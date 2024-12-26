
#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <net/if.h>
#include <net/route.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <linux/if_tun.h>
#include <fcntl.h>
#include <linux/netlink.h>

void setBaseNetAddress(char net_prefix[16], char *baseAddr)
{
    strncpy(net_prefix, baseAddr, 16);
}

// Add Gateway to the interface
int set_gateway(char *interfaceName, char *gateway)
{
    int sock_fd;
    struct rtentry rt;
    struct sockaddr_in addr;

    if ((sock_fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket failed");
        return 1;
    }

    memset(&rt, 0, sizeof(rt));
    addr.sin_family = AF_INET;
  /*set Destination addr*/
    inet_aton("0.0.0.0", &addr.sin_addr);
    memcpy(&rt.rt_dst, &addr, sizeof(struct sockaddr_in));
  /*set gateway addr*/
    inet_aton(gateway, &addr.sin_addr);
    memcpy(&rt.rt_gateway, &addr, sizeof(struct sockaddr_in));
  /*set genmask addr*/
    inet_aton("0.0.0.0", &addr.sin_addr);
    memcpy(&rt.rt_genmask, &addr, sizeof(struct sockaddr_in));
    rt.rt_dev = interfaceName;
  // rt.rt_flags = RTF_UP|RTF_GATEWAY|RTF_DEFAULT;
  /* SR: rt_flags on 16 bits but RTF_DEFAULT = 0x00010000
     * therefore doesn't lie in container -> disable it
     */
  // rt.rt_flags = RTF_GATEWAY|RTF_DEFAULT;
    rt.rt_flags = RTF_GATEWAY;

    if (ioctl(sock_fd, SIOCADDRT, &rt) < 0) {
        close(sock_fd);

        if (strstr(strerror(errno), "File exists") == NULL) {
            printf( "ioctl SIOCADDRT failed : %s\n", strerror(errno));
            return 2;
        } else { /*if SIOCADDRT error is route exist, retrun success*/
            printf(  "File Exist ...\n");
            printf(  "set_gateway OK!\n");
            return 0;
        }
    }

    close(sock_fd);
    printf( "Set Gateway OK!\n");
    printf("1");
    return 0;
}

// sets a genneric interface parameter
// (SIOCSIFADDR, SIOCSIFNETMASK, SIOCSIFBRDADDR, SIOCSIFFLAGS)
int setInterfaceParameter(char *interfaceName, char *settingAddress, int operation)
{
    int sock_fd;
    struct ifreq ifr;
    struct sockaddr_in addr;

    if ((sock_fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        printf(
              "Setting operation %d, for %s, address, %s : socket failed\n",
              operation,
              interfaceName,
              settingAddress);
        return 1;
    }

    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, interfaceName, sizeof(ifr.ifr_name) - 1);
    memset(&addr, 0, sizeof(struct sockaddr_in));
    addr.sin_family = AF_INET;
    inet_aton(settingAddress, &addr.sin_addr);
    memcpy(&ifr.ifr_ifru.ifru_addr, &addr, sizeof(struct sockaddr_in));
    // ioctl(skfd, SIOCSIFFLAGS, &ifr);
    if (ioctl(sock_fd, operation, &ifr) < 0) {
        close(sock_fd);
        printf(
              "Setting operation %d, for %s, address, %s : ioctl call failed\n",
              operation,
              interfaceName,
              settingAddress);
        return 2;
    }

    close(sock_fd);
    return 0;
}

// sets a genneric interface parameter
// (SIOCSIFADDR, SIOCSIFNETMASK, SIOCSIFBRDADDR, SIOCSIFFLAGS)
int bringInterfaceUp(int mtu, char *interfaceName, int up)
{
    int sock_fd;
    struct ifreq ifr;
    int ret, err;

    if ((sock_fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        printf( "Bringing interface UP, for %s, failed creating socket\n", interfaceName);
        return 1;
    }

    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, interfaceName, sizeof(ifr.ifr_name) - 1);
    if (up) {
            ifr.ifr_flags |= IFF_UP | IFF_NOARP | IFF_MULTICAST;

            if (ioctl(sock_fd, SIOCSIFFLAGS, (caddr_t)&ifr) == -1) {
                close(sock_fd);
                printf("Bringing interface UP, for %s, failed UP ioctl\n", interfaceName);
                return 2;
            }

            ifr.ifr_mtu = mtu;
            if (ioctl(sock_fd, SIOCSIFMTU, (caddr_t)&ifr) == -1) {
                close(sock_fd);
                printf("Bringing interface UP, for %s, failed set mtu ioctl\n", interfaceName);
                return 2;
            }

    } else {
    //        printf("desactivation de %s\n", interfaceName);
        ifr.ifr_flags &= (~IFF_UP);

        if (ioctl(sock_fd, SIOCSIFFLAGS, (caddr_t)&ifr) == -1) {
            close(sock_fd);
            printf( "Bringing interface down, for %s, failed UP ioctl\n", interfaceName);
            return 2;
        }
    }

  //   printf("UP/DOWN OK!\n");
    close(sock_fd);
    return 0;
}

// non blocking full configuration of the interface (address, and the two lest
// octets of the address)
int config_tun_test( char interfaceName[32],
                char ipAddress[20],
                char net_mask[16],
                char broadcastAddress[32])
{

    int interface_id = 0;
    int returnValue;
    int mtu = 9000;
    bringInterfaceUp(mtu, interfaceName, 0);
    // sets the machine address
    returnValue = setInterfaceParameter(interfaceName, ipAddress, SIOCSIFADDR);

    // sets the machine network mask
    if (!returnValue) {
        returnValue = setInterfaceParameter(interfaceName, net_mask, SIOCSIFNETMASK);
    }

    // sets the machine broadcast address
    if (!returnValue) {
        returnValue = setInterfaceParameter(interfaceName, broadcastAddress, SIOCSIFBRDADDR);
    }

    if (!returnValue) {
        returnValue = bringInterfaceUp(mtu, interfaceName, 1);
    }

    if (!returnValue) {
       printf(
              "Interface %s successfully configured, ip address %s, mask %s "
              "broadcast address %s\n",
              interfaceName,
              ipAddress,
              net_mask,
              broadcastAddress);
    } else {
        printf(
              "Interface %s couldn't be configured (ip address %s, mask %s "
              "broadcast address %s)\n",
              interfaceName,
              ipAddress,
              net_mask,
              broadcastAddress);
    }

    int res;
    char command_line[500];

    res = sprintf(command_line,
                  "ip rule add from %s/32 table %d && "
                  "ip rule add to %s/32 table %d && "
                  "ip route add default dev %s%d table %d",
                  ipAddress,
                  interface_id - 1 + 10000,
                  ipAddress,
                  interface_id - 1 + 10000,
                  "sdr_tun",
                  interface_id,
                  interface_id - 1 + 10000);

    if (res < 0) {
        printf( "Could not create ip rule/route commands string\n");
        return res;
    }

    system(command_line);
    return returnValue;
}

// creates the broadcast address if it wasn't set before
void createBroadcast(char *broadcastAddress)
{
    int pos = strlen(broadcastAddress) - 1;

    while (broadcastAddress[pos] != '.') {
        pos--;
    }

    broadcastAddress[++pos] = '2';
    broadcastAddress[++pos] = '2';
    broadcastAddress[++pos] = '5';
    broadcastAddress[++pos] = '\0';
}
static int tun_alloc_stress(char *dev)
{
    struct ifreq ifr;
    int fd, err;

    if ((fd = open("/dev/net/tun", O_RDWR)) < 0) {
       printf("[TUN] failed to open /dev/net/tun\n");
        return -1;
    }

    memset(&ifr, 0, sizeof(ifr));
    /* Flags: IFF_TUN   - TUN device (no Ethernet headers)
     *        IFF_TAP   - TAP device
     *
     *        IFF_NO_PI - Do not provide packet information
     */
    ifr.ifr_flags = IFF_TUN | IFF_NO_PI;

    if (*dev) {
        memcpy(ifr.ifr_name, dev, sizeof(ifr.ifr_name) - 1);
    }

    if ((err = ioctl(fd, TUNSETIFF, (void *)&ifr)) < 0) {
        close(fd);
        return err;
    }

    strcpy(dev, ifr.ifr_name);
    return fd;
}

int netlink_init_tun_stress(char ifname[64])
{
    int ret;
    int sock_fd;
    sock_fd = tun_alloc_stress(ifname);

    if (sock_fd == -1) {
       printf("TUN: Error opening socket %s (%d:%s)\n", ifname, errno, strerror(errno));
        abort();
    }

    printf( 
          "[UE %d] TUN: Opened socket %s with fd nas_sock_fd=%d\n",
          0,
          ifname,
          sock_fd);
    ret = fcntl(sock_fd, F_SETFL, O_NONBLOCK);

    if (ret == -1) {
        printf("TUN: Error fcntl (%d:%s)\n", errno, strerror(errno));

        abort();
    }
    return 1;
}

// main function
//---------------------------------------------------------------------------
int main(int argc, char **argv)
//---------------------------------------------------------------------------
{
    int c;
    char interfaceName[100];
    char ipAddress[100];
    char networkMask[100];
    char broadcastAddress[100];
    strcpy(interfaceName, "sdr_tun0");
    strcpy(ipAddress, "172.29.16.16");
    strcpy(networkMask, "255.255.255.0");
    broadcastAddress[0] = '\0';

    if (strlen(broadcastAddress) == 0) {
        strcpy(broadcastAddress, ipAddress);
        createBroadcast(broadcastAddress);
    }

    printf("Command: ifconfig %s %s networkMask %s broadcast %s\n",
           interfaceName,
           ipAddress,
           networkMask,
           broadcastAddress);
    netlink_init_tun_stress(interfaceName);
    config_tun_test(interfaceName, ipAddress, networkMask, broadcastAddress);
    while(1){
        printf("Here we should write\read bytes into\from 'sdr_tun' interface\n");
        sleep(5);
    }

    return 0;
}