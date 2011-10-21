/*
 * mt19937ar.h
 *
 * $Id: mt19937ar.h,v 1.2 2004/05/03 13:17:35 thyssen Exp $
 */

#ifndef _MT19937AR_H_
#define _MT19937AR_H_
#include <stdio.h>

void init_genrand(unsigned long s);
void mt_rand(unsigned long s);
unsigned long genrand_int31(void);
double genrand_real1(void);
double genrand_real2(void);
double genrand_real3(void);
unsigned int mt_rand_mint(unsigned int max);
unsigned long mt_rand_mlong(unsigned long max);

#define mt_rand_int  genrand_int31
#define mt_rand_r1   genrand_real1
#define mt_rand_r2   genrand_real2
#define mt_rand_r3   genrand_real3
#define mt_srand	 init_genrand

#endif
