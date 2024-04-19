## 1.1.1 utils.h

```
/*

Functions:
- SWAP8(a)
- SWAP4(a)
- SWAP2(a)
- ASSERT(cond)

Constants:
- M_2PI
- M_SQRT1_2
- M_PI
- COMPLEXFFT_BLOCKSIZE

Parameters:
- a: Input value to be swapped (for SWAP8, SWAP4, SWAP2).
- cond: Condition to be checked (for ASSERT).

*/
```

## 1.1.2 utils.cpp

```
/*
没有具体的类、结构体或函数声明，只包含了头文件的包含和nts命名空间的使用
*/
```

## 1.2.1 WaveLoader.h

```
/*
Classes:
- WaveInfo: Represents information about a Wave file, including sample frequency, sample count, number of channels, etc.
    Methods:
        - IsStreamed()
        - SampFreq()
        - SampleCount()
        - Duration()
        - NumChannels()
        - BlockAlign()
        - DataBytes()
        - ReverseBytes()
        - Read(std::istream& is)
        - sampFreq_
        - samp_count_
        - num_channels_
        - reverse_bytes_
- WaveData: Represents Wave file audio data along with sample frequency.
    Methods:
        - Read(std::istream& is)
        - Write(std::ostream& os) const
        - Data()
        - SampFreq()
        - Duration()
        - CopyFrom(const WaveData& other)
        - Clear()
        - Swap(WaveData* other)
    Members:
        - data_
        - sampFreq_


Constants:
- kWaveSampleMax

Inputs:
- std::istream& is: Input stream for reading Wave file data.

Outputs:
- std::ostream& os: Output stream for writing Wave file data.

Members:
- sampFreq_, samp_count_, num_channels_, reverse_bytes_: Members of the WaveInfo class.
- data_, sampFreq_: Members of the WaveData class.

Parameters:
- is: Input stream for Read() functions.
- os: Output stream for Write() functions.
- a: Input value for byte swapping functions.

Function:
- Read(std::istream& is), Write(std::ostream& os)
- IsStreamed(), SampFreq(), SampleCount(), Duration(), NumChannels(), BlockAlign(), DataBytes(), ReverseBytes()
- Data(), SampFreq(), Duration(), CopyFrom(const WaveData& other), Clear(), Swap(WaveData* other)

*/
```

## 1.2.2 WaveLoader.cpp

```
/*
Classes:
- WaveInfo:
    Methods:
        - Read(std::istream& is):
    Members:
        - reverse_bytes_
        - sampFreq_
        - num_channels_
        - samp_count_
    Inputs:
        - is: Input stream for reading wave header data.
    Outputs:
        - None (information stored in class members).

- WaveData: 
    Methods:
        - Read(std::istream& is)
        - Write(std::ostream& os) const
    Members:
        - data_
        - sampFreq_
    Inputs:
        - is: Input stream for reading audio data.
        - os: Output stream for writing audio data.
    Outputs:
        - None (data read or written to/from class member).

Structures:
- WaveHeaderReadGofer: 
    Members:
        - is
        - swap
        - tag[5]
    Methods:
        - Expect4ByteTag(const char* expected)
        - Read4ByteTag()
        - ReadUint32()
        - ReadUint16()
    Inputs:
        - expected: Expected 4-byte tag for validation.
    Outputs:
        - None (information stored in structure members).

Functions:
- SWAP8(a):
    Inputs:
        - a: 64-bit value to be swapped.
    Outputs:
        - None (value is swapped in place).
- SWAP4(a):
    Inputs:
        - a: 32-bit value to be swapped.
    Outputs:
        - None (value is swapped in place).
- SWAP2(a):
    Inputs:
        - a: 16-bit value to be swapped.
    Outputs:
        - None (value is swapped in place).
- ASSERT(cond): 
    Inputs:
        - cond: Condition to be checked.
    Outputs:
        - None (throws error if condition is false).
- WriteUint32(os, i):
    Inputs:
        - os: Output stream for writing.
        - i: Unsigned 32-bit integer value.
    Outputs:
        - None (value is written to output stream).
- WriteUint16(os, i):
    Inputs:
        - os: Output stream for writing.
        - i: Unsigned 16-bit integer value.
    Outputs:
        - None (value is written to output stream).

Constants:
- M_2PI
- M_SQRT1_2
- M_PI
- COMPLEXFFT_BLOCKSIZE

*/
```

## 1.3.1 FeatureWindow.h

```
/*
Classes:
- FbankComputer
- OfflineFeatureTpl

Structures:
- FrameExtractionOptions
- FeatureWindowFunction
- RandomState

Functions:
- INT32 NumFrames(INT64 num_samples, const FrameExtractionOptions& opts, bool flush)
- INT64 FirstSampleOfFrame(INT32 frame, const FrameExtractionOptions& opts)
- void Dither(XTensor waveform, float dither_value)
- void Preemphasize(XTensor waveform, float preemphCoeff)
- void ProcessWindow(const FrameExtractionOptions& opts, const FeatureWindowFunction& window_function, XTensor& window, float* log_energy_pre_window = NULL)
- void ExtractWindow(INT64 sample_offset, const XTensor wave, INT32 f, const FrameExtractionOptions& opts, const FeatureWindowFunction& window_function, XTensor& window, float* log_energy_pre_window = NULL)
- float RandUniform(struct RandomState* state = NULL)
- float RandGauss(struct RandomState* state = NULL)
- void OfflineFeatureTpl<F>
- void OfflineFeatureTpl<F>
- void OfflineFeatureTpl<F>
- void OfflineFeatureTpl<F>
- void OfflineFeatureTpl<F>
- const class WaveData& OfflineFeatureTpl<F>
- INT32 OfflineFeatureTpl<F>


Inputs:
- num_samples (INT64)
- frame (INT32)
- opts (FrameExtractionOptions)
- waveform (XTensor)
- dither_value (float)
- preemphCoeff (float)
- window_function (FeatureWindowFunction)
- sample_offset (INT64)
- wave (XTensor)
- vtln_warp (float)
- noNeedComputeSize (int)
- output (XTensor*)
- sample_freq (float)

Outputs:
- log_energy_pre_window (float*)

Members:
- sampFreq (float)
- frameShiftMs (float)
- frameLengthMs (float)
- chunkLengthMs (float)
- dither (float)
- preemphCoeff (float)
- removeDcOffset (bool)
- windowType (char[MAX_NAME_LEN])
- roundToPowerOfTwo (bool)
- blackmanCoeff (float)
- snipEdges (bool)
- allowDownsample (bool)
- allowUpsample (bool)
- maxFeatureVectors (int)
- torchPaddingLength (int)
- padMod (char[MAX_NAME_LEN])
- inputAudio (char[MAX_PATH_LEN])
- window (XTensor)

Parameters:
- state (RandomState*)

*/

```

## 1.3.2 FeatureWindow.cpp

```
/*
Classes:
- FeatureWindowFunction

Structures:
- FrameExtractionOptions

Functions:
- FeatureWindowFunction(const FrameExtractionOptions& opts)
- INT64 FirstSampleOfFrame(INT32 frame, const FrameExtractionOptions& opts)
- void Dither(XTensor* waveform, float dither_value)
- void Preemphasize(XTensor* waveform, float preemphCoeff)
- INT32 NumFrames(INT64 num_samples, const FrameExtractionOptions& opts, bool flush)
- void ProcessWindow(const FrameExtractionOptions& opts, const FeatureWindowFunction& window_function, XTensor &window, float* log_energy_pre_window)
- void ExtractWindow(INT64 sample_offset, const XTensor wave, INT32 f, const FrameExtractionOptions& opts, const FeatureWindowFunction& window_function, XTensor &window, float* log_energy_pre_window)
- signed int RoundUpToNearestPowerOfTwo(signed int n)


Inputs:
- FrameExtractionOptions
- XTensor


Members:
- window

*/

```

## 1.4.1 Fbank.h

```
/*
Classes:
- MelBanks:
    Methods:
    - InverseMelScale
    - MelScale
    - VtlnWarpFreq
    - VtlnWarpMelFreq
    - Compute
    - NumBins
    - GetCenterFreqs
    Members:
    - center_freqs_
    - bins_
    - debug_
    - htkMode_
    Inputs:
    - fft_energies
    - vtln_warp_factor
    Outputs:
    - mel_energies_out

- FbankComputer
    Methods:
    - Dim
    - NeedRawLogEnergy
    - GetFrameOptions
    - NeedLog10
    - Compute
    Members:
    - FbankOptions opts_
    - float logEnergyFloor_
    - XTensor filter
    - std::map<float, MelBanks*> mel_banks_
    - SplitRadixRealFft<float>* srfft_
    Inputs:
    - signal_raw_log_energy
    - vtln_warp
    - signal_frame
    Outputs:
    - feature

Structures:
- MelBanksOptions:
    Variables:
    - INT32 numBins
    - float lowFreq
    - float highFreq
    - float vtlnLow
    - float vtlnHigh
    - bool debugMel
    - bool htkMode
    - char customFilter[MAX_PATH_LEN]
    Functions:
    - explicit MelBanksOptions(int numBins = 23)

- FbankOptions:
    Variables:
    - FrameExtractionOptions frameOpts
    - MelBanksOptions melOpts
    - bool useEnergy
    - float energyFloor
    - bool rawEnergy
    - bool htkCompat
    - bool useLogFbank
    - bool usePower
    - bool oneSide
    Functions:
    - FbankOptions()
    - FbankOptions(S2TConfig& config)
    - explicit FbankOptions(const FbankOptions& opts)
    Inputs:
    - config
    - signal_raw_log_energy
    - vtln_warp
    - signal_frame
    Outputs:
    - feature
    - fft_energies
*/

```

## 1.4.2 Fbank.cpp

```
/*
Classes:
- XTensor:
    Members:
    - order
    - Resize(int order, int* dimSize)
    - SetData(void* data, int dimSize, int startIndex):
    - GetDim(int index)
    - Get1D(int index)
    - Set1D(float value, int index)
    - GetData(int* index, int dimSize, int startIndex)
    - Cell()

- FbankComputer:
    Members:
    - opts_
    - filter
    - mel_banks_
    - logEnergyFloor_
    - srfft_
    - ComputePowerSpectrum(XTensor* waveform)
    - Compute(float signal_raw_log_energy, float vtln_warp, XTensor* signal_frame, XTensor &feature)
    - GetMelBanks(float vtln_warp)

- MelBanks:
    Members:
    - center_freqs_
    - bins_
    - debug_
    - htkMode_
    - MelBanks(const MelBanksOptions& opts, const FrameExtractionOptions& frameOpts, float vtln_warp_factor)
    - MelBanks(const MelBanks& other)
    - Compute(const XTensor& power_spectrum, XTensor* mel_energies_out) const

Inputs:
- waveform
- signal_raw_log_energy
- vtln_warp
- signal_frame
- opts

Outputs:
- feature
- mel_energies_out

Functions:
- ComputePowerSpectrum
- Compute
- VtlnWarpFreq
- VtlnWarpMelFreq
- MelBanks
- Compute
*/

```

## 1.5.1 Fbank-function.h

```
/*
Classes:
- SplitRadixComplexFft: 
  Members:
  - N_
  - logn_
  - brseed_
  - tab_
  - temp_buffer_

  Functions:
  - SplitRadixComplexFft(N)
  - SplitRadixComplexFft(const SplitRadixComplexFft&)
  - Compute(Real*, Real*, bool)
  - Compute(Real*, bool)
  - Compute(Real*, bool, std::vector<Real>*)
  - ComputeTables()
  - ComputeRecursive(Real*, Real*, INT32)
  - BitReversePermute(Real*, INT32)
  - ~SplitRadixComplexFft()

- SplitRadixRealFft:
  Members:
  - N_

  Functions:
  - SplitRadixRealFft(N)
  - Compute(Real*, bool)
  - Compute(Real*, bool, std::vector<Real>*)

Functions:
- RealFft<T>(XTensor*, bool)
- ComplexFft<T>(XTensor*, bool, XTensor*)
- OneSidedFFT<T>(XTensor*, bool)
- OneSidedFFT(const std::vector<float>&)

Constants:
- INT_TO_int

Inputs:
- Real and imaginary parts of a signal for FFT computation.
- Array or vector of float values for one-sided FFT computation.

Outputs:
- Transformed FFT data of the input signal.

*/

```

## 1.5.2 Fbank-function.cpp

```
/*

Classes:

SplitRadixComplexFft

Parameters:
N (INT32)

Functions:
ComputeTables()

Methods:
SplitRadixComplexFft(INT32 N)
~SplitRadixComplexFft()
SplitRadixComplexFft(const SplitRadixComplexFft<Real>& other)

Members:
N_ (INT32)
logn_ (INT32)
brseed_ (INT32[])
tab_ (pointer to arrays)
*/

```

## 1.6 Fbank-function-inl.h

```
/*
None
*/
```

## 1.7 MultiThread.h

```
/*
Classes:
 - ThreadPool
   - Functions:
     - submit(Func&& func, Ts &&... params)
     - threadsNum()

   - Methods:
     - ThreadPool()
     - ThreadPool(size_t maxThreads)
     - ~ThreadPool()
     - worker()
     - joinFinishedThreads()

   - Inputs:
     - Func
     - Ts

   - Outputs:
     - std
     - size_t

   - Members:
     - quit_
     - currentThreads_
     - idleThreads_
     - maxThreads_
     - mutex_
     - cv_
     - tasks_
     - finishedThreadIDs_
     - threads_

*/
```
### 2 S2T
## 2.1.1 S2TConfig.h

```
/*
Classes:
 - S2TModelConfig
    - Parameters: fbank, nConv, convKernel, convStride
    - Functions: Load
    - Inputs: argsNum, args
    - Members: fbank, nConv, convKernel, convStride

 - WhisperDecConig
    - Parameters: task, numLanguage, language, temperature, noSpeechThreshold, logProbThreshold, compRatioThreshold, withoutTimeStamps
    - Functions: Load, InitLanguageToken
    - Constants: MAX_NAME_LEN
    - Inputs: argsNum, args
    - Members: task, numLanguage, language, temperature, noSpeechThreshold, logProbThreshold, compRatioThreshold, withoutTimeStamps

 - InferenceConfig

 - ExtractionConfig
    - Parameters: useEnergy, energyFloor, rawEnergy, htkCompat, useLogFbank, usePower, oneSide, inputAudio, sampFreq, frameShiftMs, frameLengthMs, chunkLengthMs, dither, preemphCoeff, removeDcOffset, windowType, roundToPowerOfTwo, blackmanCoeff, snipEdges, allowDownsample, allowUpsample, maxFeatureVectors, torchPaddingLength, padMod, numBins, lowFreq, highFreq, vtlnLow, vtlnHigh, debugMel, htkMode, customFilter
    - Functions: Load
    - Constants: MAX_PATH_LEN, MAX_NAME_LEN
    - Inputs: argsNum, args
    - Members: useEnergy, energyFloor, rawEnergy, htkCompat, useLogFbank, usePower, oneSide, inputAudio, sampFreq, frameShiftMs, frameLengthMs, chunkLengthMs, dither, preemphCoeff, removeDcOffset, windowType, roundToPowerOfTwo, blackmanCoeff, snipEdges, allowDownsample, allowUpsample, maxFeatureVectors, torchPaddingLength, padMod, numBins, lowFreq, highFreq, vtlnLow, vtlnHigh, debugMel, htkMode, customFilter

 - S2TConfig
    - Parameters: extractor, s2tmodel, inference, whisperdec
    - Functions: LoadFromFile, showConfig
    - Inputs: argc, argv, configFN, args
    - Members: extractor, s2tmodel, inference, whisperdec

Structures:
 - LanguageUnion
    - Parameters: languageToken, language
    - Constants: MAX_NAME_LEN
    - Members: languageToken, language
*/

```

## 2.1.2 S2TConfig.cpp
```
/*
Classes:
- S2TConfig
- S2TModelConfig
- WhisperDecConfig
- ExtractionConfig

S2TConfig:
- Parameters: argc (int), argv (const char**), MAX_PARAM_NUM (constant int)
- Functions: S2TConfig(int argc, const char** argv), LoadFromFile(const char* configFN, char** args), showConfig(), Load(int argsNum, const char** args), InitLanguageToken()
- Methods: LoadParamString(), LoadInt(), LoadString(), LoadBool(), LoadFloat()
- Constants: MAX_PARAM_NUM
- Inputs: argc, argv, configFN, argsNum, args
- Members: model (S2TModelConfig object), s2tmodel (S2TModelConfig object), common (CommonConfig object), training (TrainingConfig object), inference (InferenceConfig object), whisperdec (WhisperDecConfig object), extractor (ExtractionConfig object)

S2TModelConfig:
- Parameters: argsNum (int), args (const char**)
- Functions: Load(int argsNum, const char** args)
- Methods: LoadInt(), LoadString()
- Inputs: argsNum, args
- Members: fbank (int)

WhisperDecConfig:
- Parameters: argsNum (int), args (const char**)
- Functions: Load(int argsNum, const char** args), InitLanguageToken()
- Methods: LoadInt(), LoadString(), LoadFloat(), LoadBool()
- Inputs: argsNum, args
- Members: numLanguage (int), task (char*), language (struct Language), temperature (float), noSpeechThreshold (float), logProbThreshold (float), compRatioThreshold (float), withoutTimeStamps (bool)

ExtractionConfig:
- Parameters: argsNum (int), args (const char**)
- Functions: Load(int argsNum, const char** args)
- Methods: LoadBool(), LoadFloat(), LoadString(), LoadInt()
- Inputs: argsNum, args
- Members: useEnergy (bool), energyFloor (float), rawEnergy (bool), htkCompat (bool), useLogFbank (bool), usePower (bool), oneSide (bool), inputAudio (char*), sampFreq (float), frameShiftMs (float), frameLengthMs (float), chunkLengthMs (float), dither (float), preemphCoeff (float), removeDcOffset (bool), windowType (char*), roundToPowerOfTwo (bool), blackmanCoeff (float), snipEdges (bool), allowDownsample (bool), allowUpsample (bool), maxFeatureVectors (int), torchPaddingLength (int), padMod (char*), numBins (int), lowFreq (float), highFreq (float), vtlnLow (float), vtlnHigh (float), debugMel (bool), htkMode (bool), customFilter (char*)

Structures:
- Language:
  - Parameters: languageToken (int), language (char*)
  - Members: languageToken (int), language (char*)
*/

```

## 2.2.1 S2TDataset.h

```
/*
Classes:
 - S2TDataSetBase
    - Functions: LoadBatchToBuf, Init
    - Methods: MaxAudioLen, MaxTgtLen, MaxSrcLen, SortByAudioLengthAscending, SortByTgtLengthAscending, SortBySrcLengthAscending, SortByAudioLengthDescending, SortByTgtLengthDescending, SortBySrcLengthDescending, ClearBuf, LoadSample, GetBatchSimple
    - Members: fc, wc, sc, bufIdx, buf, config

Structures:
 - TripleSample
    - Parameters: XTensor* a, IntList* s, IntList* t, int myKey
    - Functions: constructor, destructor
    - Inputs: audio sequence (a), source sequence (s), target sequence (t), bucket key (myKey)
    - Members: index, bucketKey, fLen, audioSeq, audioPath, srcSeq, tgtSeq

*/


```

## 2.2.2 S2TDataset.cpp

```
/*
Classes:
 - TripleSample
   - Constructor with XTensor, IntList*, IntList*, int
   - Constructor with string, int, IntList*, IntList*, int
   - Destructor
   - Members: index, audioSeq, audioPath, srcSeq, tgtSeq, fLen, bucketKey

 - S2TDataSetBase
   - Constructor
   - Destructor
   - MaxAudioLen(int, int)
   - MaxTgtLen(int, int)
   - MaxSrcLen(int, int)
   - SortByAudioLengthAscending()
   - SortByTgtLengthAscending()
   - SortBySrcLengthAscending()
   - SortByAudioLengthDescending()
   - SortByTgtLengthDescending()
   - SortBySrcLengthDescending()
   - ClearBuf()
   - Members: fc, wc, sc, bufIdx, config, buf

Constants:
 - MAX
*/
```

## 2.3.1 S2TEncoder.h

```
/*
Classes:
 - S2TAttEncoder
   - Methods:
     - S2TAttEncoder()
     - ~S2TAttEncoder()
     - void InitModel(S2TConfig& config)
     - XTensor RunFastPreNorm(XTensor& input, XTensor* mask)
   - Constants:
     - Extractor* extractor
     - XTensor posEmbeddingBase
   - Members:
     - Extractor* extractor
     - XTensor posEmbeddingBase

*/

```

## 2.3.2 S2TEncoder.cpp

```
/*
Classes:
 - S2TAttEncoder
   - Parameters:
     - devID (int)
     - extractor (Extractor*)
     - selfAtts (Attention*)
     - ffns (FFN*)
     - attLayerNorms (LayerNorm*)
     - fnnLayerNorms (LayerNorm*)
     - encoderLayerNorm (LayerNorm*)
     - useHistory (bool)
     - history (LayerHistory*)
     - dropoutP (float)
     - embDim (int)
     - finalNorm (bool)
     - ignored (int)
     - nlayer (int)
     - preLN (bool)
     - vSize (int)
     - isTraining (bool)

   - Methods:
     - S2TAttEncoder()
     - ~S2TAttEncoder()
     - InitModel(S2TConfig& config)
     - RunFastPreNorm(XTensor& input, XTensor* mask)

*/

```

## 2.4.1 S2TModel.h

```
/*
Classes:
 - S2TModel
   - Parameters: devID, config, encoder, decoder, outputLayer
   - Functions: S2TModel(), ~S2TModel(), InitModel(), MakeS2TMaskEnc(), MakeS2TTriMaskDecInference(), TestDumpParams()
   - Methods: GetIntConfigs(), GetBoolConfigs(), GetFloatConfigs(), GetParams(), LoadFromFile()
   - Inputs: config (S2TConfig*), input (XTensor&), inputEnc (XTensor&), inputDec (XTensor&), paddingEnc (XTensor&), paddingDec (XTensor&), maskEnc (XTensor&), maskDec (XTensor&), maskEncDec (XTensor&), params (XTensor*), batchSize (int), length (int), file (FILE*)
   - Members: devID (int), config (S2TConfig*), encoder (S2TAttEncoder*), decoder (AttDecoder*), outputLayer (OutputLayer*)
*/

```

## 2.4.2 S2TModel.cpp

```
/*
Classes:
 - S2TModel
    - Parameters:
        - devID
        - config
        - encoder
        - decoder
        - outputLayer
        - intConfig
        - boolConfig
        - floatConfig
        - modelFile
        - myConfig
        - startT
        - elapsed
        - params
        - padding2
        - maskEnc
        - maskEncDec
    - Functions:
        - S2TModel()
        - ~S2TModel()
        - GetIntConfigs()
        - GetBoolConfigs()
        - GetFloatConfigs()
        - InitModel(S2TConfig* myConfig)
        - GetParams(TensorList& list)
        - LoadFromFile(FILE* file)
        - TestDumpParams(XTensor* params)
        - MakeS2TMaskEnc(XTensor& paddingEnc, XTensor& maskEnc)
        - MakeS2TTriMaskDecInference(int batchSize, int length)

 - S2TAttEncoder: 
    - Parameters:
        - nConv
        - useHistory
        - nlayer
        - selfAtts
        - ffns
        - attLayerNorms
        - fnnLayerNorms
        - encoderLayerNorm
        - posEmbeddingBase
    - Functions:
        - InitModel(S2TConfig& config)。

 - AttDecoder:
    - Parameters:
        - nlayer
        - selfAtts
        - enDeAtts
        - ffns
        - selfAttLayerNorms
        - enDeAttLayerNorms
        - ffnLayerNorms
        - decoderLayerNorm
        - posEmbeddingBase
    - Functions:
        - InitModel(S2TConfig& config)

 - OutputLayer
    - Parameters:
        - w
    - Functions:
        - InitModel(S2TConfig& config)
*/
```

## 2.5.1 S2TVocab.h

```
/*
Classes:
 - S2TVocab
    - Parameters: 
    - sosID, 
    - eosID, 
    - padID, 
    - unkID, 
    - noTimeStamps, 
    - noSpeech, 
    - startPrev,
    - startLM, 
    - langIDs, 
    - tStampIDs, 
    - taskIDs, 
    - vocabSize, 
    - token2id, 
    - id2token, 
    - timestampMask, 
    - nonTimestampMask, 
    - contentMask
    
    - Functions: 
    - constructor, 
    - de-constructor, 
    - SetSpecialID, 
    - InitMask, 
    - Load, 
    - Save, 
    - CopyFrom, 
    - ShowVocab, 
    - Test, 
    - DecodingWord

*/

```

## 2.5.2 S2TVocab.cpp

```
/*
Classes:
-S2TVocab:
 - Members: 
    - int sosID, eosID, padID, unkID, vocabSize
    - int* langIDs, tStampIDs, taskIDs
    - unordered_map<string, int> token2id, id2token
 - Methods:
    - decodeEscapedString(const std::string& escaped)
    - SetSpecialID(int sos, int eos, int pad, int unk, int numLanguage)
    - InitMask(int devID)
    - S2TVocab()
    - ~S2TVocab()
    - Load(const string& vocabFN)
    - Save(const string& vocabFN)
    - CopyFrom(const S2TVocab& v)
    - ShowVocab()
    - DecodingWord(vector<int>* tokensId): std:
    - Test()
 - Constants:
    - X_INT
 - Inputs:
    - std::string escaped (decodeEscapedString)
    - int sos, eos, pad, unk, numLanguage, devID (SetSpecialID, InitMask)
    - const string& vocabFN (Load, Save)
    - const S2TVocab& v (CopyFrom)
    - vector<int>* tokensId (DecodingWord)
    - None (ShowVocab, Test)
 - Outputs:
    - std::string decoded string (decodeEscapedString)
 - Functions:
    - decodeEscapedString(const std::string& escaped)
    - SetSpecialID(int sos, int eos, int pad, int unk, int numLanguage)
    - InitMask(int devID)
    - Load(const string& vocabFN)
    - Save(const string& vocabFN)
    - CopyFrom(const S2TVocab& v)
    - ShowVocab()
    - DecodingWord(vector<int>* tokensId)
    - Test()
*/

```
### 3 generate
## 3.1.1 Generator.h

```
/*
Classes:
 - Generator
   - Parameters: model, batchLoader, searcher, config, vocab, outputBuf, oft
   - Functions: Init, Generate, Interact, TestInference
   - Methods: DecodingBatch
   - Inputs: S2TConfig*, S2TModel*, bool
   - Outputs: bool
*/

```

## 3.1.2 Generator.cpp

```
/*
Classes:
 - Generator
   - Parameters: model, batchLoader, searcher, config, vocab, outputBuf, oft
   - Functions: Init, Generate, Interact, TestInference
   - Methods: DecodingBatch
   - Inputs: S2TConfig*, S2TModel*, bool (for Init), char*, char*, bool (for Interact)
   - Outputs: bool (for Generate, TestInference)

*/

```

## 3.2.1 S2TGeneratorDataset.h

```
/*
Classes:
 - S2TGeneratorDataset
   - Members:
     - appendEmptyLine (bool)
     - emptyLines (IntList)
     - srcVocab (S2TVocab)
     - tgtVocab (S2TVocab)
     - ifp (istream*)
   - Methods:
     - IsEmpty()
     - Init(S2TConfig& myConfig, bool notUsed)
     - LoadSample()
     - LoadSample(XTensor* s)
     - LoadSample(string s, int n_frames)
     - LoadSample(XTensor* s, string line)
     - GetBatchSimple(XList* inputs, XList* info)
     - LoadBatchToBuf()
     - S2TGeneratorDataset()
     - ~S2TGeneratorDataset()
*/

```

## 3.2.2 S2TGeneratorDataset.cpp

```
/*
Classes:
 - S2TGeneratorDataset:
   - Parameters: 
    - config (S2TConfig), 
    - ifp (ifstream*), 
    - appendEmptyLine (bool), 
    - bufIdx (int), 
    - buf (XList*), 
    - srcVocab (Vocab), 
    - tgtVocab (Vocab)
   - Functions: 
    - Init(S2TConfig&, bool), 
    - IsEmpty(), 
    - ~S2TGeneratorDataset(), 
    - LoadBatchToBuf(), 
    - GetBatchSimple(XList*, XList*), 
    - LoadSample(XTensor*), 
    - LoadSample(string, int), 
    - LoadSample(XTensor*, string), 
    - LoadSample(), 
    - ClearBuf(), 
    - SortByAudioLengthDescending()
   - Inputs: 
    - config (S2TConfig), 
    - ifp (ifstream*), 
    - appendEmptyLine (bool), 
    - bufIdx (int), 
    - buf (XList*), 
    - srcVocab (Vocab), 
    - tgtVocab (Vocab), 
    - XList*
   - Members: 
    - config (S2TConfig*), 
    - ifp (ifstream*), 
    - appendEmptyLine (bool), 
    - bufIdx (int), 
    - buf (XList*), 
    - srcVocab (Vocab), 
    - tgtVocab (Vocab)
 - TripleSample:
  - Members: 
    - audioPath (string), 
    - audioSeq (XTensor*), 
    - fLen (int), 
    - index (int)
  - Functions: 
    - TripleSample(XTensor*), 
    - TripleSample(string, int), 
    - TripleSample(XTensor*, 
    - IntList*), 
    - ~TripleSample()
*/

```

## 3.3.1 S2TSearcher.h

```
/*
Classes:
- S2TGreedySearch
  - Members:
    - maxLen
    - batchSize
    - endSymbols
    - endSymbolNum
    - startSymbols
    - startSymbolNum
    - suppressSymbols
    - suppressSymbolNum
    - scalarMaxLength
    - vocab
    - withoutTimeStamps
  - Methods:
    - Init
    - InitStartSymbols
    - InitPromptSymbols
    - InitSuppressSymbols
    - IsEnd
    - Search
    - Suppress
    - Predict

- S2TPredictor
  - Members:
    - beamSize
    - batchSize
    - m
    - s
    - endSymbols
    - endSymbolNum
    - startSymbols
    - startSymbolNum
    - suppressSymbols
    - suppressSymbolNum
    - vocab
    - withoutTimeStamps
  - Methods:
    - Init
    - Read
    - Create
    - Predict
    - Suppress

- S2TBeamSearch
  - Members:
    - alpha
    - predictor
    - maxLen
    - beamSize
    - batchSize
    - fullHypos
    - endSymbols
    - endSymbolNum
    - startSymbols
    - startSymbolNum
    - suppressSymbols
    - suppressSymbolNum
    - scalarMaxLength
    - isEarlyStop
    - aliveStatePids
    - aliveSentList
    - needReorder
    - vocab
    - withoutTimeStamps
  - Methods:
    - Init
    - Prepare
    - Score
    - Expand
    - Collect
    - FillHeap
    - Dump
    - IsEnd
    - Search

*/

```

## 3.3.2 S2TSearcher.cpp

```
/*
Classes:
 - S2TBeamSearch
    - Parameters: batch size, beam size, alpha, scalar max length, max length, without time stamps, end symbols, end symbol num, start symbols, start symbol num, suppress symbols, suppress symbol num, vocab
    - Functions: InitSuppressSymbols, InitPromptSymbols, InitStartSymbols, Init, Prepare, Score, Expand, Collect, FillHeap, Dump, IsEnd, Search
    - Methods: Init, Prepare, Score, Expand, Collect, FillHeap, Dump, IsEnd, Search
    - Constants: MIN_HEAP
    - Inputs: input, padding, outputs, score, model, top, token, beam, reorderState, prev, modelScore, prob, probPath, probPathPrev, idRef, modelScoreRef, probRef, probPathRef, predictionRef, endMark, id, encoding, maskEnc, aliveState, inputEnc, paddingEnc, maskDec, decoding
    - Outputs: output, fullHypos, tgt, aliveSentList, aliveStatePids, suppressTokens, promptTokens, suppressSymbols, startSymbols, endSymbols, states, aliveState, reorderState, modify, timestampLogprob, maxTextLogprob, logits, timestampLogprob, maxTextLogprob, prob, first, inputDec
    - Members: batch size, beam size, alpha, scalar max length, max length, without time stamps, end symbols, end symbol num, start symbols, start symbol num, suppress symbols, suppress symbol num, vocab, m, s, batchSize, stateNum, nstep, isStart, probPath, endMark, modelScore, prob, probPath, prediction, last, pid, isCompleted, model, top, prediction, prob, probPath, aliveState, reorderState, output, fullHypos, tgt, aliveSentList, aliveStatePids, suppressTokens, promptTokens, suppressSymbols, startSymbols, endSymbols, states, modify, timestampLogprob, maxTextLogprob, logits, timestampLogprob, maxTextLogprob, prob, first, inputDec

 - StateBundle
    - Parameters: state num, nstep, isStart, probPath, endMark, modelScore, prediction, last, pid, isCompleted, states, top
    - Functions: Create, MakeStates, MakeS2TMaskEnc, Read, Suppress, Predict
    - Methods: Create, MakeStates, MakeS2TMaskEnc, Read, Suppress, Predict
    - Inputs: input, batch size, beam size, end symbols, end symbol num, start symbols, start symbol num, suppress symbols, suppress symbol num, vocab, model, top, token, inputEnc, paddingEnc, aliveState, reorderState, next, encoding, maskDec, decoding, aliveState, reorderState, reorderStateCPU, reorderStateCPU, reorderStateCPU, reorderStateCPU, reorderStateCPU, reorderStateCPU, reorderStateCPU, reorderStateCPU, reorderStateCPU, reorderStateCPU, reorderStateCPU, reorderStateCPU, reorderStateCPU, reorderStateCPU, reorderStateCPU, reorderStateCPU
    - Outputs: modify, input, input, inputDec, maskDec, decoding, output, output, output, output, output, output, output, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, prob, first, prob
    - Members: state num, nstep, isStart, probPath, endMark, modelScore, prediction, last, pid, isCompleted, states, top, batch size, beam size, end symbols, end symbol num, start symbols, start symbol num, suppress symbols, suppress symbol num, vocab, m, s, batchSize, stateNum, nstep, isStart, probPath, endMark, modelScore, prob, probPath, prediction, last, pid, isCompleted, model, top, prediction, prob, probPath, aliveState, reorderState, output, fullHypos, tgt, aliveSentList, aliveStatePids, suppressTokens, promptTokens, suppressSymbols, startSymbols, endSymbols, states, modify, timestampLogprob, maxTextLogprob, logits, timestampLogprob, maxTextLogprob, prob, first, inputDec

Constants:
 - MIN_HEAP
    - Value

Public Functions/Methods:
 - InitSuppressSymbols
 - InitPromptSymbols
 - InitStartSymbols
 - Init
 - Prepare
 - Score
 - Expand
 - Collect
 - FillHeap
 - Dump
 - IsEnd
 - Search
 - Create
 - MakeStates
 - MakeS2TMaskEnc
 - Read
 - Suppress
 - Predict

Public Inputs/Outputs/Members:
 - batchSize
 - beamSize
 - alpha
 - scalarMaxLength
 - maxLength
 - withoutTimeStamps
 - endSymbols
 - endSymbolNum
 - startSymbols
 - startSymbolNum
 - suppressSymbols
 - suppressSymbolNum
 - vocab
 - m
 - s
 - batch size
 - stateNum
 - nstep
 - isStart
 - probPath
 - endMark
 - modelScore
 - prob
 - probPath
 - prediction
 - last
 - pid
 - isCompleted
 - model
 - top
 - prediction
 - prob
 - probPath
 - aliveState
 - reorderState
 - output
 - fullHypos
 - tgt
 - aliveSentList
 - aliveStatePids
 - suppressTokens
 - promptTokens
 - suppressSymbols
 - startSymbols
 - endSymbols
 - states
 - modify
 - timestampLogprob
 - maxTextLogprob
 - logits
 - timestampLogprob
 - maxTextLogprob
 - prob
 - first
 - inputDec
*/


```

### 4 submodel
## 4.1.1 Extractor.h

```
/*
Classes:
 - Extractor
    - Parameters: 
     - isTraining (bool), 
     - devID (int), 
     - inSize (int), 
     - hSize (int), 
     - nConv (int), 
     - convKernels (vector<int>), 
     - convStrides (vector<int>), 
     - kernels (XTensor*), 
     - biases (XTensor*), 
     - dropoutP (DTYPE)
    - Functions: 
     - SetTrainingFlag(bool), 
     - InitModel(S2TConfig&), 
     - Make(XTensor&)
    - Methods: 
     - Constructor, 
     - De-constructor
*/
```

## 4.1.2 Extractor.cpp

```
/*
Classes:
 - Extractor
   - Parameters: 
    - isTraining, 
    - devID, 
    - inSize, 
    - hSize, 
    - nConv, 
    - kernels, 
    - biases, 
    - dropoutP
   - Functions: 
    - SetTrainingFlag, 
    - InitModel, 
    - Make
   - Methods: 
    - Extractor (constructor), 
    - ~Extractor (destructor)
   - Inputs: 
    - config (S2TConfig), 
    - input (XTensor)
   - Outputs: 
    - outFeature (XTensor)
   - Members: 
    - isTraining, 
    - devID, 
    - inSize, 
    - hSize, 
    - nConv, 
    - kernels, 
    - biases, 
    - dropoutP
   - Constants: 
    - X_FLOAT (data type)
 - S2TConfig
   - Parameters: 
    - training.isTraining, 
    - common.devID, 
    - s2tmodel.nConv, 
    - s2tmodel.convKernel, 
    - s2tmodel.convStride, 
    - s2tmodel.fbank, 
    - model.encEmbDim
   - Members: 
    - training.isTraining, 
    - common.devID, 
    - s2tmodel.nConv, 
    - s2tmodel.convKernel, 
    - s2tmodel.convStride, 
    - s2tmodel.fbank, 
    - model.encEmbDim
 - XTensor
   - Parameters: 
    - shape, 
    - dataType, 
    - devID
   - Functions: 
    - Conv1DBias, 
    - GELU, 
    - InitTensor3D, 
    - InitTensor1D
   - Methods: 
    - InitTensor3D, 
    - InitTensor1D
   - Inputs: 
    - input, 
    - kernels[i], 
    - biases[i], 
    - convStrides[i]
   - Outputs: 
    - outFeature
   - Members: 
    - shape, 
    - dataType, 
    - devID
   - Constants: 
    - X_FLOAT (data type)

*/

```
