
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

/* common RX and TX streaming params */
struct stream_cfg {
	long long bw_hz; // Analog banwidth in Hz
	long long fs_hz; // Baseband sample rate in Hz
	long long lo_hz; // Local oscillator frequency in Hz
    long long buffer_size;
    const char *rfport; // Port name
};

int main(){
    std::cout << "Hello, world!" << std::endl;

    // Конфиг. параметры "потоков"
	struct stream_cfg rxcfg;
	struct stream_cfg txcfg;

    // RX stream config
	rxcfg.bw_hz = MHZ(2);   // 2 MHz rf bandwidth
	rxcfg.fs_hz = MHZ(2.5);   // 2.5 MS/s rx sample rate
	rxcfg.lo_hz = GHZ(1); // 2.5 GHz rf frequency
	rxcfg.rfport = "A_BALANCED"; // port A (select for rf freq.)
    rxcfg.buffer_size = pow(2, 16); // размер буфера в сэмплах

    // TX stream config
    txcfg.bw_hz = MHZ(1.5); // 1.5 MHz rf bandwidth
    txcfg.buffer_size = pow(2, 16); // размер буфера в сэмплах

    // TX stream config
	txcfg.bw_hz = MHZ(2); // 1.5 MHz rf bandwidth
	txcfg.fs_hz = MHZ(2.5);   // 2.5 MS/s tx sample rate
	txcfg.lo_hz = GHZ(1); // 2.5 GHz rf frequency
	txcfg.rfport = "A"; // port A (select for rf freq.)


    // Initialize IIO context
    ctx = iio_create_context(NULL, "ip:192.168.3.1");
    if(!ctx){
        std::cerr << "Unable to create IIO context addr: " << "ip:192.168.3.1" << std::endl;
        return 1;
    } else {
        std::cout << "IIO context created successfully, addr: " << "ip:192.168.3.1" << std::endl;
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



    rxstream = iio_buffer_create_stream(rxbuf, 4, rxcfg.buffer_size);
    txstream = iio_buffer_create_stream(txbuf, 4, txcfg.buffer_size);
    // RX and TX sample size
	size_t rx_sample_sz, tx_sample_sz;
    rx_sample_sz = iio_device_get_sample_size(rx_dev, rxmask);
	tx_sample_sz = iio_device_get_sample_size(tx_dev, txmask);
    printf("* rx_sample_sz = %d [bytes]\n",rx_sample_sz);
    printf("* tx_sample_sz = %d [bytes]\n",tx_sample_sz);

    struct iio_block *txblock, *rxblock;

    int32_t counter = 0;
    // Открываем файл для записи данных
    std::ofstream outfile("rx_signal.txt", std::ios::out);

    if (!outfile.is_open()) {
        std::cerr << "Unable to open file for writing" << std::endl;
        //return;
    }
    while (1)
    {
        counter++;
        int16_t *p_dat, *p_end;
        ptrdiff_t p_inc;
        uint32_t samples_cnt = 0;

        
        rxblock = iio_buffer_create_block(rxbuf, rxcfg.buffer_size);
        /* READ: Get pointers to RX buf and read IQ from RX buf port 0 */
		p_inc = rx_sample_sz;
		p_end = static_cast<int16_t *>(iio_block_end(rxblock));
        //printf("iio_block_first = %d, iio_block_end = %d, p_inc = %d\n", iio_block_first(rxblock, rx0_i), p_end, p_inc);
		for (p_dat = static_cast<int16_t *> (iio_block_first(rxblock, rx0_i)); p_dat < p_end;
		     p_dat += p_inc / sizeof(*p_dat)) {
			/* Example: swap I and Q */
			int16_t i = p_dat[0];
			int16_t q = p_dat[1];
            outfile << i << ", " << q << std::endl;
		}
        iio_block_enqueue(rxblock, 0, false);
        iio_buffer_enable(rxbuf);
        iio_block_dequeue(rxblock, false);

        txblock = iio_buffer_create_block(txbuf, txcfg.buffer_size);
        /* WRITE: Get pointers to TX buf and write IQ to TX buf port 0 */
        p_inc = tx_sample_sz;
        p_end = static_cast<int16_t *>(iio_block_end(txblock));
        for (p_dat = static_cast<int16_t *>(iio_block_first(txblock, tx0_i)); p_dat < p_end;
            p_dat += p_inc / sizeof(*p_dat)) {
            if(counter % 10 == 0){
                p_dat[0] = 10000; /* Real (I) */
                p_dat[1] = 10000; /* Imag (Q) */
            } else {
                p_dat[0] = 10; /* Real (I) */
                p_dat[1] = 10; /* Imag (Q) */
            }
        }

        iio_block_enqueue(txblock, 0, false);
        iio_buffer_enable(txbuf);
        iio_block_dequeue(txblock, false);

        // printf("samples_cnt = %d\n", samples_cnt);
        
    }

    return 0;
}