/*******************************************************************************
* Copyright (c) 2021 Cadence Design Systems, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to use this Software with Cadence processor cores only and
* not with any other processors and platforms, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

******************************************************************************/
#include "xa_type_def.h"
#include "common.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"

#if XCHAL_HAVE_HIFI1
//output: output_inv_sqrt (ae_int32x2), output_shift (int)
//input:  input (ae_int32x2) , reverse_shift (int)
#define GET_INV_SQRT_QUANTIZED_MULTIPLIER_EXP(output_inv_sqrt, output_shift, input, reverse_shift){\
  ae_int32x2 CT_Q31_minus_1, CT_Q31, CT_Q29, CT_ONE;\
  CT_Q31_minus_1 = AE_MOVDA32(Q31_minus_1);\
  CT_Q31 = AE_MOVDA32(Q31);\
  CT_Q29 = AE_MOVDA32(Q29);\
  CT_ONE = AE_MOVDA32(1);\
\
  xtbool2 b1, b2;\
  b1 = AE_LE32(input, CT_ONE);\
\
  if(AE_MOVAB2(b1))\
  {\
    output_inv_sqrt = AE_MOV32(CT_Q31_minus_1);\
    output_shift = 0;\
  }\
  else\
  {\
    output_shift = 11;\
    b2 = AE_LT32(input, CT_Q29);\
    while(!AE_MOVAB2(b2))\
    {\
      input = AE_SRAI32(input, 2);\
      ++output_shift;\
      b2 = AE_LT32(input, CT_Q29);\
    }\
\
    int max_left_shift_bits, max_left_shift_bit_pairs, left_shift_bit_pairs;\
    max_left_shift_bits = AE_NSA32_L(input);\
    max_left_shift_bit_pairs = max_left_shift_bits / 2;\
    left_shift_bit_pairs = max_left_shift_bit_pairs - 1;\
    output_shift -= left_shift_bit_pairs;\
    input = AE_SLAA32(input, (2*left_shift_bit_pairs));\
\
    ae_int32x2 fixedpoint_input, fixedpoint_half_input, fixedpoint_half_three, x, x2, x3, y1, y2;\
    fixedpoint_input = AE_SRAI32(input, 1);\
    fixedpoint_half_input = AE_SRAI32R(fixedpoint_input, 1);\
    fixedpoint_half_three = AE_MOVDA32(FIXED_POINT_HALF_THREE);\
    x = AE_MOVDA32(FIXED_POINT_ONE);\
\
    int i = 0;\
    for(i=0; i<5; i++)\
    {\
      x2 = AE_MULFP32X2RS(x, x);\
      x3 = AE_MULFP32X2RS(x2, x);\
      x3 = AE_SLAI32S(x3, 6);\
\
      y1 = AE_MULFP32X2RS(fixedpoint_half_three, x);\
      y2 = AE_MULFP32X2RS(fixedpoint_half_input, x3);\
\
      x = AE_SUB32S(y1, y2);\
      x = AE_SLAI32S(x, 3);\
    }\
\
    ae_int32x2 fixedpoint_half_sqrt_2;\
    fixedpoint_half_sqrt_2 = AE_MOVDA32(FIXED_POINT_HALF_SQRT_2);\
    output_inv_sqrt = AE_MULFP32X2RS(x, fixedpoint_half_sqrt_2);\
    if(output_shift < 0)\
    {\
      output_inv_sqrt = AE_SLAA32S(output_inv_sqrt, -output_shift);\
      output_shift = 0;\
    }\
    output_shift *= reverse_shift;\
\
  }\
}

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out, inp1, inp2, multiplier, left_shift, right_shift) \
{\
  inp1 = AE_SLAA32S(inp1, left_shift); \
  inp2 = AE_SLAA32S(inp2, left_shift); \
  inp1 = AE_MULFP32X2RAS(inp1, AE_NEG32(AE_MOVDA32(multiplier))); \
  inp2 = AE_MULFP32X2RAS(inp2, AE_NEG32(AE_MOVDA32(multiplier))); \
  inp1 = AE_MULFP32X2RS(inp1, right_shift); \
  inp2 = AE_MULFP32X2RS(inp2, right_shift); \
  out = AE_SAT16X4(inp1, inp2); \
}

static const int Q31_minus_1 = 0x7fffffff;
static const int Q31         = 0x80000000;
static const int Q29         = 0x20000000;
static const int FIXED_POINT_HALF_THREE = 0x18000000;
static const int FIXED_POINT_ONE = 0x10000000;
static const int FIXED_POINT_HALF_SQRT_2 = 0x5a82799a;

WORD32 xa_nn_l2_norm_asym8s_asym8s(WORD8 *p_out,
                      const WORD8 *p_inp,
                            WORD32 zero_point,
                            WORD32 num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND(((zero_point < -128) || (zero_point > 127)), -1);
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

  WORD8 *p_in  = (WORD8 *)p_inp;
  WORD8 *p_o   = (WORD8 *)p_out;

  int output_scale = 7;
  int reverse_shift = -1;

  int i = 0;
  int rem_length = (num_elm & 3);

  ae_valign align_src, align_dst;
  align_src = AE_LA64_PP(p_in);
  align_dst = AE_ZALIGN64();

  ae_int16x4 m1, z_16x4;
  ae_int16x4 z10;
  ae_int32x2 acc;
  ae_int64 acc_0 = 0;
  z_16x4 = AE_MOVDA16(zero_point);

  for(i=0; i<(num_elm >> 2); i++)
  {
    AE_LA8X4S_IP(m1, align_src, p_in);
    z10 = AE_SUB16(m1, z_16x4);

    AE_MULAAAAQ16(acc_0, z10, z10);
  }

  // remainder loop
  for(i=0; i<(rem_length); i++)
    {
      AE_L8S_IP(m1, p_in, sizeof(WORD8));
      z10 = AE_SUB16(m1, z_16x4);
      z10 = AE_SEL16_6543(AE_MOV16(0), z10);

      AE_MULAAAAQ16(acc_0, z10, z10);
    }

  acc = AE_TRUNCA32X2F64S(acc_0, acc_0, 32);

  ae_int32x2 inv_l2norm_multiplier;
  int inv_l2norm_shift;
  GET_INV_SQRT_QUANTIZED_MULTIPLIER_EXP(inv_l2norm_multiplier, inv_l2norm_shift, acc, reverse_shift);

  int shift = inv_l2norm_shift + output_scale;
  int left_shift  = shift<0 ? 0 : shift;

  int right_shift = shift>0 ? 0 :-shift;
  right_shift = (0XFFFFFFFF << (31 - right_shift));

  ae_int32x2 x32, x10;

  p_in  = (WORD8 *)p_inp;
  align_src = AE_LA64_PP(p_in);

  ae_int16x4 one_16x4 = AE_MOVDA16(1);
  for(i=0; i<(num_elm >> 2); i++)
  {
    AE_LA8X4S_IP(m1, align_src, p_in);
    z10 = AE_SUB16(m1, z_16x4);

    AE_MUL16X4(x32, x10, z10, one_16x4);

    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(z10, x32, x10, inv_l2norm_multiplier, left_shift, right_shift);

    m1 = AE_SAT8S(z10);

    AE_SA8X4U_IP(m1, align_dst, (ae_int32*)p_o);
  }
  AE_SA64POS_FP(align_dst, p_o);

  // remainder loop
  for(i=0; i<(rem_length); i++)
  {
    AE_L8S_IP(m1, p_in, sizeof(WORD8));
    z10 = AE_SUB16(m1, z_16x4);

    AE_MUL16X4(x32, x10, z10, one_16x4);

    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(z10, x32, x10, inv_l2norm_multiplier, left_shift, right_shift);

    m1 = AE_SAT8S(z10);

    AE_S8_0_IP(m1, p_o, sizeof(WORD8));
   }

  return 0;
}
#else
WORD32 xa_nn_l2_norm_asym8s_asym8s(WORD8 *p_out,
                      const WORD8 *p_inp,
                            WORD32 zero_point,
                            WORD32 num_elm)
{
  return -1;
}
#endif
