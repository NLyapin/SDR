
#include <iio/iio.h>
#include <iio/iio-debug.h>

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <iostream>
#include <fstream>

/* helper macros */
#define MHZ(x) ((long long)(x*1000000.0 + .5))
#define GHZ(x) ((long long)(x*1000000000.0 + .5))


/* IIO structs required for streaming */
static struct iio_context *ctx   = NULL;
static struct iio_channel *rx0_i = NULL;
static struct iio_channel *rx0_q = NULL;
static struct iio_channel *tx0_i = NULL;
static struct iio_channel *tx0_q = NULL;
static struct iio_buffer  *rxbuf = NULL;
static struct iio_buffer  *txbuf = NULL;
static struct iio_stream  *rxstream = NULL;
static struct iio_stream  *txstream = NULL;
static struct iio_channels_mask *rxmask = NULL;
static struct iio_channels_mask *txmask = NULL;



/* cleanup and exit */
static void shutdown(void)
{
	printf("* Destroying streams\n");
	if (rxstream) {iio_stream_destroy(rxstream); }
	if (txstream) { iio_stream_destroy(txstream); }

	printf("* Destroying buffers\n");
	if (rxbuf) { iio_buffer_destroy(rxbuf); }
	if (txbuf) { iio_buffer_destroy(txbuf); }

	printf("* Destroying channel masks\n");
	if (rxmask) { iio_channels_mask_destroy(rxmask); }
	if (txmask) { iio_channels_mask_destroy(txmask); }

	printf("* Destroying context\n");
	if (ctx) { iio_context_destroy(ctx); }
}

struct sigaction old_action;

void sigint_handler(int sig_no)
{
    printf("CTRL-C pressed\n");
    sigaction(SIGINT, &old_action, NULL);
    shutdown();
    exit(0);
}

/* common RX and TX streaming params */
struct stream_cfg {
	long long bw_hz; // Analog banwidth in Hz
	long long fs_hz; // Baseband sample rate in Hz
	long long lo_hz; // Local oscillator frequency in Hz
	const char* rfport; // Port name
};

int main(){
    std::cout << "Hello, world!" << std::endl;
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = &sigint_handler;
    sigaction(SIGINT, &action, &old_action);

    // Конфиг. параметры "потоков"
	struct stream_cfg rxcfg;
	struct stream_cfg txcfg;

    // RX stream config
	rxcfg.bw_hz = MHZ(1);   // 2 MHz rf bandwidth
	rxcfg.fs_hz = MHZ(2);   // 2.5 MS/s rx sample rate
	rxcfg.lo_hz = MHZ(900); // 2.5 GHz rf frequency
	rxcfg.rfport = "A_BALANCED"; // port A (select for rf freq.)

	// TX stream config
	txcfg.bw_hz = MHZ(1); // 1 MHz rf bandwidth
	txcfg.fs_hz = MHZ(2);   // 2.5 MS/s tx sample rate
	txcfg.lo_hz = MHZ(1000); // 2.5 GHz rf frequency
	txcfg.rfport = "A"; // port A (select for rf freq.)


    // Initialize IIO context
    ctx = iio_create_context(NULL, "ip:192.168.2.1");
    if(!ctx){
        std::cerr << "Unable to create IIO context addr: " << "ip:192.168.2.1" << std::endl;
        return 1;
    } else {
        std::cout << "IIO context created successfully, addr: " << "ip:192.168.2.1" << std::endl;
    }

// The ad9361-phy driver entirely Controls the AD9361.
// The cf-ad9361-lpc is the ADC/RX capture driver that controls the RX DMA and the RX HDL core.
// The cf-ad9361-dds-core-lpc is the DAC/TX output driver that controls the TX DMA and the TX HDL core, including the DDS.

    printf("* Инициализация AD9361 устройств\n");
    struct iio_device *phy_dev;
    struct iio_device *tx_dev;
    struct iio_device *rx_dev;
    
    tx_dev =    iio_context_find_device(ctx, "cf-ad9361-dds-core-lpc");
    rx_dev =    iio_context_find_device(ctx, "cf-ad9361-lpc");
    phy_dev  =  iio_context_find_device(ctx, "ad9361-phy");



    printf("* Настройка параметров %s канала AD9361 \n", "TX");
	struct iio_channel *tx_chn = NULL;
    tx_chn = iio_device_find_channel(phy_dev, "voltage0", true);
    const struct iio_attr *tx_rf_port_attr = iio_channel_find_attr(tx_chn, "rf_port_select");
    iio_attr_write_string(tx_rf_port_attr, txcfg.rfport);
    const struct iio_attr *tx_bw_attr = iio_channel_find_attr(tx_chn, "rf_bandwidth");
    iio_attr_write_longlong(tx_bw_attr, txcfg.bw_hz);
    const struct iio_attr *tx_fs_attr = iio_channel_find_attr(tx_chn, "sampling_frequency");
    iio_attr_write_longlong(tx_fs_attr, txcfg.fs_hz);
    printf("* Настройка частоты опорного генератора (lo, local oscilator)  %s \n", "TX");
    struct iio_channel *tx_lo_chn = NULL;
    tx_lo_chn = iio_device_find_channel(phy_dev, "altvoltage1", true);
    const struct iio_attr *tx_lo_attr = iio_channel_find_attr(tx_lo_chn, "frequency");
    iio_attr_write_longlong(tx_lo_attr, txcfg.lo_hz);

    printf("* Настройка параметров %s канала AD9361 \n", "RX");
    struct iio_channel *rx_chn = NULL;
    rx_chn = iio_device_find_channel(phy_dev, "voltage0", false);
    const struct iio_attr *rx_rf_port_attr = iio_channel_find_attr(rx_chn, "rf_port_select");
    iio_attr_write_string(rx_rf_port_attr, rxcfg.rfport);
    const struct iio_attr *rx_bw_attr = iio_channel_find_attr(rx_chn, "rf_bandwidth");
    iio_attr_write_longlong(rx_bw_attr, rxcfg.bw_hz);
    const struct iio_attr *rx_fs_attr = iio_channel_find_attr(rx_chn, "sampling_frequency");
    iio_attr_write_longlong(rx_fs_attr, rxcfg.fs_hz);
    printf("* Настройка частоты опорного генератора (lo, local oscilator)  %s \n", "RX");
    struct iio_channel *rx_lo_chn = NULL;
    rx_lo_chn = iio_device_find_channel(phy_dev, "altvoltage0", true);
    const struct iio_attr *rx_lo_attr = iio_channel_find_attr(rx_lo_chn, "frequency");
    iio_attr_write_longlong(rx_lo_attr, rxcfg.lo_hz);

    printf("* Инициализация потоков I/Q %s канала AD9361 \n", "TX");
    tx0_i = iio_device_find_channel(tx_dev, "voltage0", true);
    tx0_q = iio_device_find_channel(tx_dev, "voltage1", true);

    rx0_i = iio_device_find_channel(rx_dev, "voltage0", false);
    rx0_q = iio_device_find_channel(rx_dev, "voltage1", false);

    rxmask = iio_create_channels_mask(iio_device_get_channels_count(rx_dev));
    if (!rxmask) {
		fprintf(stderr, "Unable to alloc RX channels mask\n");
	}

	txmask = iio_create_channels_mask(iio_device_get_channels_count(tx_dev));
	if (!txmask) {
		fprintf(stderr, "Unable to alloc TX channels mask\n");
	}

    printf("* Enabling IIO streaming channels\n");
	iio_channel_enable(rx0_i, rxmask);
	iio_channel_enable(rx0_q, rxmask);
	iio_channel_enable(tx0_i, txmask);
	iio_channel_enable(tx0_q, txmask);

    printf("* Creating non-cyclic IIO buffers with 1 MiS\n");
	rxbuf = iio_device_create_buffer(rx_dev, 0, rxmask);
    txbuf = iio_device_create_buffer(tx_dev, 0, txmask);

    rxstream = iio_buffer_create_stream(rxbuf, 4, pow(2, 14));
    txstream = iio_buffer_create_stream(txbuf, 4, pow(2, 14));
    // RX and TX sample size
	size_t rx_sample_sz, tx_sample_sz;
    rx_sample_sz = iio_device_get_sample_size(rx_dev, rxmask);
	tx_sample_sz = iio_device_get_sample_size(tx_dev, txmask);
    printf("* rx_sample_sz = %l\n",rx_sample_sz);
    printf("* tx_sample_sz = %l\n",tx_sample_sz);

    const struct iio_block *txblock, *rxblock;

    // Открываем файл для записи данных
    std::ofstream outfile("1_rx_signal.txt", std::ios::out);
    int16_t tx_q[330] = {16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,-16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384};
    int16_t tx_i[330] = {16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, -16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384};
    int16_t rx_i[1000000];
    int16_t rx_q[1000000];
    int16_t tx_file_i[1000000];
    int16_t tx_file_q[1000000];
    memset (rx_i, 0, 1000000);
    memset (rx_q, 0, 1000000);
    memset (tx_file_i, 0, 1000000);
    memset (tx_file_q, 0, 1000000);
    for (int j = 0; j < 10; j++){
        printf("i = %d\n", tx_q[j]);
    }

    int32_t counter = 0;
    int32_t i = 0, j= 0;
    while (1)
    {
        int16_t *p_dat, *p_end;
		ptrdiff_t p_inc;
        uint32_t samples_cnt = 0;

        rxblock = iio_stream_get_next_block(rxstream);
        txblock = iio_stream_get_next_block(txstream);

        /* WRITE: Get pointers to TX buf and write IQ to TX buf port 0 */
        if(counter % 2 == 0){
            p_inc = tx_sample_sz;
            p_end = static_cast<int16_t *>(iio_block_end(txblock));
            int counter_i = 0;
            for (p_dat = static_cast<int16_t *>(iio_block_first(txblock, tx0_i)); p_dat < p_end;
                p_dat += p_inc / sizeof(*p_dat))
            {
                // if(counter_i >= 1000 && counter_i < 1330){
                //     p_dat[0] = (tx_i[counter_i]);               // pow(2, 12); /* Real (I) */
                //     p_dat[1] = (tx_q[counter_i]); 
                //     // tx_file_i[j] = p_dat[0];
                //     // tx_file_q[j] = p_dat[1];              // pow(2, 12);; /* Imag (Q) */
                // }
                // else
                // {
                //     p_dat[0] = 10; /* Real (I) */
                //     p_dat[1] = 10; /* Imag (Q) */
                //     // tx_file_i[j] = p_dat[0];
                //     // tx_file_q[j] = p_dat[1];
                // }
                if(counter_i < 330){
                    p_dat[0] = (tx_i[counter_i]) / 4;               // pow(2, 12); /* Real (I) */
                    p_dat[1] = (tx_q[counter_i]) / 4; 
                } else {
                    counter_i = 0;
                }
                counter_i++;
                j++ ;
            }
        } else {
            p_inc = tx_sample_sz;
            p_end = static_cast<int16_t *>(iio_block_end(txblock));
            for (p_dat = static_cast<int16_t *>(iio_block_first(txblock, tx0_i)); p_dat < p_end;
                p_dat += p_inc / sizeof(*p_dat))
            {
                p_dat[0] = 10; /* Real (I) */
                p_dat[1] = 10; /* Imag (Q) */
                // tx_file_i[j] = p_dat[0];
                // tx_file_q[j] = p_dat[1];
                j++ ;
            }

        }

        // /* READ: Get pointers to RX buf and read IQ from RX buf port 0 */
		// p_inc = rx_sample_sz;
		// p_end = static_cast<int16_t *>(iio_block_end(rxblock));
        // printf("iio_block_first = %d, iio_block_end = %d, p_inc = %d\n", iio_block_first(rxblock, rx0_i), p_end, p_inc);
		// for (p_dat = static_cast<int16_t *> (iio_block_first(rxblock, rx0_i)); p_dat < p_end;
		//      p_dat += p_inc / sizeof(*p_dat)) {
		// 	/* Example: swap I and Q */
		// 	// int16_t i = p_dat[0];
		// 	// int16_t q = p_dat[1];
        //     rx_i[i] = p_dat[0];
        //     rx_q[i] = p_dat[1];
        //     samples_cnt++;
        //     i++;
        // }
        // printf("samples_cnt = %d\n", samples_cnt);
        // printf("i = %d\n", i);


        printf("counter = %d\n", counter);
        counter++;
    }
    shutdown();
    // for (int j = 0; j < 1000000; j++){
    //     // printf("rx_i[i] = %d\n", rx_i[j]);
    //     // printf("rx_q[i] = %d\n", rx_q[j]);
    //     outfile << rx_i[j] << ", " << rx_q[j] << std::endl;
    // }
    // std::ofstream outfile_2("1_tx_signal.txt", std::ios::out);
    // for (int j = 0; j < 1000000; j++){
    //     // printf("rx_i[i] = %d\n", rx_i[j]);
    //     // printf("rx_q[i] = %d\n", rx_q[j]);
    //     outfile_2 << tx_file_i[j] << ", " << tx_file_q[j] << std::endl;
    // }
    return 0;
}