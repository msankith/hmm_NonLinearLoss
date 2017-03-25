/***********************************************************************/
/*                                                                     */
/*   svm_struct_api.c                                                  */
/*                                                                     */
/*   Definition of API for attaching implementing SVM learning of      */
/*   structures (e.g. parsing, multi-label classification, HMM)        */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cctype> //tolower
#include <iostream>
#include <fstream>
#include <sstream> //istringstream
#include <iomanip>
#include <string>
#include <algorithm> //transform()
using namespace std;
#include <ext/hash_map> //this location is compiler-dependent
using __gnu_cxx::hash; //__gnu_cxx is where gcc sticks nonstandard STL stuff
using __gnu_cxx::hash_map;
#include <boost/tuple/tuple.hpp>
using boost::tuple;
#include "svm_struct/svm_struct_common.h"
#include "svm_struct_api.h"

/*
   define an assertion handler for when a BOOST assertion gets triggered and we want to be able to trace it upward in gdb
   (set gdb to break inside the function below)
 */
namespace boost
{
	void assertion_failed(char const * expr, char const * function, char const * file, long line)
	{
		printf("boost assertion failed: %s at %s(%ld) (%s)\n", expr, file, line, function);
		printf("\n");
	}
}

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

	template <class T>
T sqr(T t)
{
	return t * t;
}

/********** debug-printing ************/

void printSentence(PATTERN x)
{
	for(unsigned int i = 0; i < x.getLength(); i++)
		printf("%s ", x.getToken(i).getString().c_str());
	printf("\n");
}

void printLabelSeq(LABEL y)
{
	for(unsigned int i = 0; i < y.getLength(); i++)
		printf("%s ", getTagByID(y.getTag(i)).c_str());
	printf("\n");
}

/**************** tags ****************/

namespace
{
	class hashString
	{
		public:

			hashString() {}

			size_t operator () (const string& s) const
			{
				return hasher(s.c_str());
			}

		private:

			static const hash<const char*> hasher;
	};
	const hash<const char*> hashString::hasher = hash<const char*>();

	hash_map<tagID, tag> idToTagMap;
	hash_map<string, tagID, hashString> tagToIDMap;

	/*
	   during classification, we read in the training-set tags from the model, then
	   the test-set tags from the input; we want to register the first set but not the second,
	   so provide a flag
	 */
	bool registryWritable = true;

}

void setTagRegistryWritable(bool w)
{
	registryWritable = w;
}

/*
   if t is in the map,
   return a newly assigned unique tag ID
 */
tagID registerTag(const tag& t)
{
	hash_map<string, tagID, hashString>::iterator i = tagToIDMap.find(t);
	if(i != tagToIDMap.end()) //tag has been registered
		return (*i).second;
	else if(registryWritable) //tag has not been registered
	{
		idToTagMap[idToTagMap.size()] = t;
		tagToIDMap[t] = idToTagMap.size() - 1;
		return idToTagMap.size() - 1;
	}
	else //tag has not been registered, but registry is read-only
		return -1; //wraps to UINT_MAX
}

/*
   return the number of tags that have been registered
   (registering is done while reading input)
 */
unsigned int getNumTags()
{
	return idToTagMap.size();
}

const tag& getTagByID(tagID id) throw(invalid_argument)
{
	if(idToTagMap.find(id) == idToTagMap.end()) throw invalid_argument("getTagByID(): unknown ID");
	return idToTagMap[id];
}

/************* class token ************/

token::token()
{
	initFeatures();
}

token::token(const string& s) : str(s)
{
	initFeatures();
}

token::token(const token& t) : str(t.str)
{
	features = t.features;
}

token::~token()
{
	//features will delete itself
}

/*
   initialize the features map/list

   should only be called from a constructor
 */
void token::initFeatures()
{
	features = shared_ptr<SVECTOR>(new SVECTOR);
	features->words = (WORD*)my_malloc(sizeof(WORD));
	features->words[0].wnum = 0;
	//svmlight yells if userdefined is NULL; it must have at least one element
	features->userdefined = (char*)my_malloc(sizeof(char));
	features->userdefined[0] = 0;
	features->next = NULL;
	features->factor = 1;
}

const token& token::operator = (const token& t)
{
	str = t.str;
	features = t.features;
	return *this;
}

/************* class label ************/

bool label::operator == (const label& l) const
{
	if(getLength() != l.getLength()) return false;
	for(unsigned int i = 0; i < getLength(); i++)
		if(getTag(i) != l.getTag(i))
			return false;
	return true;
}

/********** class strMatcher **********/

/*
   auxiliary to read_struct_examples()
 */
inline strMatcher match(const string& s) {return strMatcher(s);}

/*
   auxiliary to read_struct_examples(): try to match a string literal in an input stream

   the stream may be partially read if an error occurs
 */
istream& operator >> (istream& in, const strMatcher& m)
{
	if(!in) return in;
	for(unsigned int i = 0; i < m.str.length(); i++)
	{
		if(in.peek() != m.str[i])
		{
			in.setstate(ios_base::failbit); //set failure
			return in;
		}
		in.get(); //extract one character
	}
	return in;
}

/**************************************/

void        svm_struct_learn_api_init(int argc, char* argv[])
{
	/* Called in learning part before anything else is done to allow
	   any initializations that might be necessary. */
}

void        svm_struct_learn_api_exit()
{
	/* Called in learning part at the very end to allow any clean-up
	   that might be necessary. */
}

void        svm_struct_classify_api_init(int argc, char* argv[])
{
	/* Called in prediction part before anything else is done to allow
	   any initializations that might be necessary. */
}

void        svm_struct_classify_api_exit()
{
	/* Called in prediction part at the very end to allow any clean-up
	   that might be necessary. */
}

/**************************************/

/*
   this function gets called by both the learning and prediction modules

   automatically generate the feature vector filename from the POS filename: POS_BASE.ext -> POS_BASE_feats.dat

   NOTE we want this to be called before init_struct_model() so we'll have defined the input language size
 */
SAMPLE      read_struct_examples(const char *filename, STRUCT_LEARN_PARM *sparm)
{
	/*
	   if we're reading the training set, fSS hasn't been set; if we're on classification,
	   it was set when we read in the model
	 */
	bool onClassification = (sparm->featureSpaceSize != 0);

	/* Reads struct examples and returns them in sample. The number of
	   examples must be written into sample.n */
	SAMPLE   sample;

	//holding space until we allocate sample.examples; note the shared_ptr default ctor gives a null pointer
	vector<shared_ptr<vector<token> > > tokens;
	vector<shared_ptr<vector<tagID> > > tagIDs;

	ifstream infile(filename);
	if(!infile)
	{
		fprintf(stderr, "read_struct_examples(): can't open '%s' for reading; exiting\n", filename);
		exit(-1);
	}

	unsigned int lineNum = 0;
	string line, comment, _tag, word;
	unsigned int exNum, exIndex, featNum, maxFeatNumFound = 0;
	double featVal;
	while(getline(infile, line, '\n') && line.length() > 0) //an empty line ends input
	{
#define PARSE_ERROR(infoDesc, lineNo)\
		{\
			fprintf(stderr, "parse error reading %s on line %u of '%s'", infoDesc, lineNo, filename);\
			exit(-1);\
		}
		//if there's a comment on the line, remove it into another string
		size_t commentIndex = line.find("#", line.find_first_of("1234567890")); //a # before the feature list must be a word; don't remove it
		if(commentIndex != string::npos)
		{
			comment = line.substr(commentIndex + 1);
			line = line.substr(0, commentIndex);
		}
		else comment = "";
		//parse tag
		istringstream instr(line);
		if(!(instr >> _tag >> match(" qid:") >> exNum >> match(".") >> exIndex)) PARSE_ERROR("token info", lineNum);
		//resize temporary storage of tokens and tags
		if(tokens.size() < exNum) //input example and token numbers start at 1
		{
			tokens.resize(exNum);
			tagIDs.resize(exNum);
		}
		if(tokens[exNum - 1].get() == NULL)
		{
			tokens[exNum - 1] = shared_ptr<vector<token> >(new vector<token>);
			tagIDs[exNum - 1] = shared_ptr<vector<tagID> >(new vector<tagID>);
		}
		if(tokens[exNum - 1]->size() < exIndex)
		{
			tokens[exNum - 1]->resize(exIndex);
			tagIDs[exNum - 1]->resize(exIndex);
		}
		(*tagIDs[exNum - 1])[exIndex - 1] = registerTag(_tag); //returns a new id only if the tag hasn't been seen before
		//parse features
		SVECTOR& features = (*tokens[exNum - 1])[exIndex - 1].getFeatureMap();
		unsigned int numFeats = 0;
		while(instr >> featNum >> match(":") >> featVal)
		{
			if(onClassification) //avoid features with higher numbers than what we saw during training
			{
				if(featNum <= sparm->featureSpaceSize)
				{
					features.words = (WORD*)realloc(features.words, ++numFeats * sizeof(WORD));
					features.words[numFeats - 1].wnum = featNum; //feature numbers start at 1 in the input
					features.words[numFeats - 1].weight = featVal;
				}
			}
			else
			{
				features.words = (WORD*)realloc(features.words, ++numFeats * sizeof(WORD));
				features.words[numFeats - 1].wnum = featNum; //feature numbers start at 1 in the input
				features.words[numFeats - 1].weight = featVal;
				if(featNum > maxFeatNumFound) maxFeatNumFound = featNum;
			}
		}
		features.words = (WORD*)realloc(features.words, ++numFeats * sizeof(WORD));
		features.words[numFeats - 1].wnum = 0; //signal to end word list
		if(instr.bad()) PARSE_ERROR("features", lineNum); //read error, as opposed to just reaching end of line
		//parse the comment (first word, if any, is interpreted as the token string; rest is ignored)
		size_t wordStart = comment.find_first_not_of(" \t\n\r");
		if(wordStart != string::npos) //the comment contains non-whitespace
		{
			comment = comment.substr(wordStart, comment.find_last_not_of(" \t\n\r") + 1); //trim whitespace
			istringstream incomment(comment);
			if(incomment >> word) (*tokens[exNum - 1])[exIndex - 1].setString(word);
		}
		lineNum++;
#undef PARSE_ERROR
	}
	infile.close();

	if(!onClassification) //if during training, figure out the feature space size
	{
		if(maxFeatNumFound == 0)
		{
			fprintf(stderr, "read_struct_examples(): fishy input: no features found; exiting\n");
			exit(-1);
		}
		sparm->featureSpaceSize = maxFeatNumFound; //feature numbers start at 1
	}

	sample.n = tokens.size();
	sample.examples = new EXAMPLE[sample.n]; //initialize the PATTERNs and LABELs from our temporary storage
	for(unsigned int i = 0; i < tokens.size(); i++)
	{
		sample.examples[i].x.setEmissionsVector(tokens[i]);
		sample.examples[i].y.setTagsVector(tagIDs[i]);
	}
	return(sample);
}

/*
   this is called BEFORE init_struct_constraints() but AFTER read_struct_examples()
 */
void init_struct_model(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, LEARN_PARM* lparm, KERNEL_PARM* kparm)
{
	/* Initialize structmodel sm. The weight vector w does not need to be
	   initialized, but you need to provide the maximum size of the
	   feature space in sizePsi. This is the maximum number of different
	   weights that can be learned. Later, the weight vector w will
	   contain the learned weights for the model. */

	/*
	   for an HMM, depends on the sizes of phi(X) and Y, the feature space and the label set
	 */
	sm->sizePsi = getNumTags() * (getNumTags() + sparm->featureSpaceSize);
}

CONSTSET    init_struct_constraints(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
	/* Initializes the optimization problem. Typically, you do not need
	   to change this function, since you want to start with an empty
	   set of constraints. However, if for example you have constraints
	   that certain weights need to be positive, you might put that in
	   here. The constraints are represented as lhs[i]*w >= rhs[i]. lhs
	   is an array of feature vectors, rhs is an array of doubles. m is
	   the number of constraints. The function returns the initial
	   set of constraints. */
	CONSTSET c;
	long     sizePsi=sm->sizePsi;
	long     i;
	WORD     words[2];

	if(1) { /* normal case: start with empty set of constraints */
		c.lhs=NULL;
		c.rhs=NULL;
		c.m=0;
	}
	else { /* add constraints so that all learned weights are
		  positive. WARNING: Currently, they are positive only up to
		  precision epsilon set by -e. */
		c.lhs=(DOC**)my_malloc(sizeof(DOC *)*sizePsi);
		c.rhs=(double*)my_malloc(sizeof(double)*sizePsi);
		for(i=0; i<sizePsi; i++) {
			words[0].wnum=i+1;
			words[0].weight=1.0;
			words[1].wnum=0;
			/* the following slackid is a hack. we will run into problems
			   if we have more than MAX_NUM_EXAMPLES slack sets (ie examples).
			   MAX_NUM_EXAMPLES is defined in svm_struct_api_types.h */
			c.lhs[i]=create_example(i,0,MAX_NUM_EXAMPLES+i,1,create_svector(words,"",1.0));
			c.rhs[i]=0.0;
		}
	}
	return(c);
}

/*
   return the index into a feature vector that denotes the y1 -> y2 transition,
   with an offset of 1 to work with svmlight
 */
inline unsigned int get_transition_feature_id(tagID y1, tagID y2)
{
	return y1 * getNumTags() + y2 + 1;
}

/*
   return the index into a feature vector that denotes the start of the output features for tag y,
   with an offset of 1 to work with svmlight
 */
unsigned int get_output_feature_start_id(tagID y, STRUCT_LEARN_PARM* sparm)
{
	static const unsigned int psqr = sqr(getNumTags()) + 1;
	return psqr + sparm->featureSpaceSize * y;
}

/*
   auxiliary to classify_struct_example(): return the log-probability, according to weight vector w,
   of the state transition y1 -> y2
 */
inline double get_transition_probability(const double* w, tagID y1, tagID y2)
{
	return w[get_transition_feature_id(y1, y2)];
}

/*
   auxiliary to classify_struct_example(): return the log-probability, according to weight vector w,
   of state y outputting a token with x's feature vector
 */
inline double get_output_probability(const double* w, tagID y, const token& x, STRUCT_LEARN_PARM* sparm)
{
	//we want the dot product of x's features with the appropriate subvector of w
	const unsigned int startIndex = get_output_feature_start_id(y, sparm);
	return x.dotProduct(&w[startIndex - 1]); //the feature numbers in x start at 1
}

vector<vector<double> > V;
vector<vector<unsigned int> > P;
LABEL       classify_struct_example(PATTERN x, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
	/* Finds the label yhat for pattern x that scores the highest
	   according to the linear evaluation function in sm, especially the
	   weights sm->w. The returned label is taken as the prediction of sm
	   for the pattern x. The weights correspond to the features defined
	   by psi() and range from index 1 to index sm->sizePsi. If the
	   function cannot find a label, it shall return an empty label as
	   recognized by the function empty_label(y). */
	LABEL y;
	int t=x.getLength(),m=getNumTags();
	V.resize( t , vector<double>( m , -2 ) );
	P.resize( t , vector<unsigned int>( m , 0 ) );
	/* use Viterbi to calculate, in order, each token's most likely state */

	static double* stateProbabilities[2] = {NULL, NULL}; //one for the current tag position and one for the previous
	static bool init = true;
	if(init)
	{
		stateProbabilities[0] = new double[getNumTags()];
		stateProbabilities[1] = new double[getNumTags()];
		init = false;
	}
	bool vecnum; //which of the two is the current 'current' vector

	double maxProb = -1;
	unsigned int maxIndex;
	/* initial probability P(x_0 | y_0 = y) for all y */
	vecnum = 0;
	for(unsigned int i = 0; i < getNumTags(); i++)
	{
		V[0][i]=stateProbabilities[vecnum][i] = get_output_probability(sm->w, (tagID)i, x.getToken(0), sparm);
		if(stateProbabilities[vecnum][i] > maxProb)
		{
			maxProb = stateProbabilities[vecnum][i];
			maxIndex = i;
		}
	}

	vector<vector<tagID> > mostLikelyPaths; //from index (j - 1, i) we can trace back the most likely path ending at state i at position j
	/* recursion: find argmax(y) {P(x_i = x'_i | y_i) P(y_i = y | y_i-1)} */
	double tempProb;
	for(unsigned int j = 1; j < x.getLength(); j++) /* loop over words in the sentence */
	{
		vecnum = !vecnum;
		mostLikelyPaths.push_back(vector<tagID>());
		maxProb = -1;
		//loop over the tag in the current spot
		for(unsigned int i = 0; i < getNumTags(); i++)
		{
			stateProbabilities[vecnum][i] = -1;
			double outputProb = get_output_probability(sm->w, i, x.getToken(j), sparm);
			//loop over the tag in the previous spot
			for(unsigned int k = 0; k < getNumTags(); k++)
			{
				//add log-"probabilities" to get a comparison equivalent to multiplying probabilities
				tempProb = stateProbabilities[!vecnum][k]							//value of previous subsequence
					+ get_transition_probability(sm->w, k, i)			//transition cost
					+ outputProb;												//output cost
				if(tempProb > stateProbabilities[vecnum][i])
				{
					stateProbabilities[vecnum][i] = tempProb;
					maxIndex = k;

				}
			}
			V[j][i]=stateProbabilities[vecnum][i];
			P[j][i]=maxIndex;
			mostLikelyPaths.back().push_back((tagID)maxIndex); //push a reference to the previous label in the most likely path to this one
		}
	}

	//find the final state whose max-prob sequence has highest probability
	/*
	cout<<endl;
	  for(unsigned int a=0;a<t;a++)
	  {
	  for(unsigned int b=0;b<m;b++)
	  {
	  cout<<V[a][b]<<" ";
	  }
	  cout<<endl;
	  for(unsigned int b=0;b<m;b++)
	  {
	  cout<<P[a][b]<<" ";
	  }
	  cout<<endl;
	  } 
	*/
	maxIndex = 0;
	maxProb = stateProbabilities[vecnum][0];
	for(unsigned int i = 1; i < getNumTags(); i++)
		if(stateProbabilities[vecnum][i] > maxProb)
		{
			maxProb = stateProbabilities[vecnum][i];
			maxIndex = i;
		}
	//build y in reverse by looking backward through the table to find the max-prob path
	y.setLength(x.getLength());
	y.setTag(y.getLength() - 1, (tagID)maxIndex);
	for(int j = x.getLength() - 2; j > -1; j--)
	{
		y.setTag(j, mostLikelyPaths[j][maxIndex]);
		maxIndex = mostLikelyPaths[j][maxIndex];
	}

	return(y);
}

LABEL       find_most_violated_constraint_slackrescaling(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
	/* Finds the label ybar for pattern x that that is responsible for
	   the most violated constraint for the slack rescaling
	   formulation. It has to take into account the scoring function in
	   sm, especially the weights sm.w, as well as the loss
	   function. The weights in sm.w correspond to the features defined
	   by psi() and range from index 1 to index sm->sizePsi. Most simple
	   is the case of the zero/one loss function. For the zero/one loss,
	   this function should return the highest scoring label ybar, if
	   ybar is unequal y; if it is equal to the correct label y, then
	   the function shall return the second highest scoring label. If
	   the function cannot find a label, it shall return an empty label
	   as recognized by the function empty_label(y). */
	LABEL ybar;

	/* insert your code for computing the label ybar here */
	fprintf(stderr, "Error: find_most_violated_constraint_slackrescaling() shouldn't be called (not used); exiting\n");
	exit(-1);

	return(ybar);
}
vector<vector<vector<double> > > C; // erstwhile V ... Cost values
vector<vector<vector<unsigned int> > > B; // back ptrs ... path
LABEL       find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
	/* Finds the label ybar for pattern x that that is responsible for
	   the most violated constraint for the margin rescaling
	   formulation. It has to take into account the scoring function in
	   sm, especially the weights sm->w, as well as the loss
	   function. The weights in sm->w correspond to the features defined
	   by psi() and range from index 1 to index sm->sizePsi. Most simple
	   is the case of the zero/one loss function. For the zero/one loss,
	   this function should return the highest scoring label ybar, if
	   ybar is unequal y; if it is equal to the correct label y, then
	   the function shall return the second highest scoring label. If
	   the function cannot find a label, it shall return an empty label
	   as recognized by the function empty_label(y). */
	LABEL ybar;
	//cout << "Entering find_most_violated_constraint ......... " << endl;

	int t=x.getLength(),m=getNumTags();
	C.resize(t,vector<vector<double> >(m,vector<double>(t+1,-100)));
	B.resize(t,vector<vector<unsigned int> >(m,vector<unsigned int>(t+1,0)));
	/* use Viterbi to calculate the cost for each possible state at each position in the input in turn */

	// static double* stateCosts[2] = {NULL, NULL}; //one for the current tag position and one for the previous
	/* static bool init = true;
	   if(init)
	   {
	   stateCosts[0] = new double[getNumTags()];
	   stateCosts[1] = new double[getNumTags()];
	   init = false;
	   }
	 */
	// bool vecnum; //which cost vector is acting as the 'current' one

	//calculate costs for the first position
	// vecnum = 0;
	//cout << "Hi 1 Enter ..." << endl;
	for(unsigned int j = 0; j < getNumTags(); j++)
		/*
		   calculate the total additional cost of adding this tag to the sequence
		   (note we don't subtract w * psi(x, y), since that's the same for every ybar)
		 */
		// stateCosts[vecnum][j] = ((j != y.getTag(0)) ? 1 : 0)
		C[0][j][0] = ((j != y.getTag(0)) ? -100 : 0)
			//mislabeling cost (loss)
			+ get_output_probability(sm->w, (tagID)j, x.getToken(0), sparm);		//output cost
	//cout << "Hi 1 Exit ..." << endl;
	// vector<vector<tagID> > mostCostlyPaths; //from index (j - 1, i) we can trace back the most likely path ending at state i at position j
	double tempCost, outputProb;
	unsigned int maxIndex;
	//cout << "Hi 2 Enter ..." << endl;
	for(unsigned int i = 1; i < x.getLength(); i++) /* loop over words in the sentence */
	{
		// vecnum = !vecnum;
		// mostCostlyPaths.push_back(vector<tagID>());
		/* calculate the cost of labeling x_i with postag j */
		//run through tags at present position
		//cout << "Hi 3 Enter ..." << endl;
		for(unsigned int j = 0; j < getNumTags(); j++) // state
		{
			outputProb = get_output_probability(sm->w, j, x.getToken(i), sparm); //probability that x[i] is output from state j
			//run through tags at previous position
			double curr_max = -100; unsigned int curr_max_index = 0;
			//cout << "Hi 4 Enter ..." << endl;
			for (unsigned int err = 0; err <= i; err++) 
			{
				if(j == y.getTag(i)) 
				{
					// C[i][j][err] = max over m_p { C[i-1][m_p][err] + w . phi }
					//cout << "Hi 5 Enter ..." << endl;
					for(unsigned int k = 0; k < getNumTags(); k++) 
					{			
						curr_max = C[i-1][k][err]						
							//cost of previous subsequence
							//+ ((j != y.getTag(i)) ? 1 : 0) 						//mislabeling cost (loss)
							+ get_transition_probability(sm->w, k, j)
							//transition cost
							+ outputProb;
						//output cost
						if(k == 0 || curr_max > C[i][j][err])
						{
							C[i][j][err] = curr_max;
							curr_max_index = k;
						}

					}
					//cout << "Hi 5 Exit ..." << endl;
				} // end if			
				else 
				{

					//double curr_max = -100; unsigned int curr_max_index = 0;
					//cout << "Hi 6 Enter ..." << endl;
					for(unsigned int k = 0; k < getNumTags(); k++) 
					{	
						if(err == 0) {
							// curr_max = C[i-1][k][0] + get_transition_probability(sm->w, k, j)
							//transition cost
							// + outputProb;
							curr_max = -100;
							continue;
						}		
						curr_max = C[i-1][k][err-1]						
							//cost of previous subsequence
							+ ( sqrt(err) - sqrt(err-1) ) 						//mislabeling cost (loss)
							+ get_transition_probability(sm->w, k, j)
							//transition cost
							+ outputProb;
						//output cost
						if(k == 0 || curr_max > C[i][j][err])
						{
							C[i][j][err] = curr_max;
							curr_max_index = k;
						}

					} // end k
					//cout << "Hi 6 Exit ..." << endl;

				} // end else
			B[i][j][err] = curr_max_index;
			} // end err for
			//cout << "Hi 4 Exit ..." << endl;
		} // end j 
		//cout << "Hi 3 Exit ..." << endl;
		// mostCostlyPaths.back().push_back((tagID)maxIndex); //push a reference to the previous tag in this tag's most costly path
		
	} // end i
	//cout << "Hi 2 Exit ..." << endl;


//find the last-position tag with the highest-cost path
// double maxCost = stateCosts[vecnum][0];
unsigned int maxI = 0;
double maxC = -100;
int maxErr = 0;
//cout << "Hi 7 Enter ..." << endl;
for(unsigned int j = 0; j < getNumTags(); j++)
{
	//cout << "Hi 8 Enter ..." << endl;
	for(int err = 0; err <= t; err++)
	{
		
		if(C[t-1][j][err] > maxC)
		{
			maxC = C[t-1][j][err];
			maxI = j;
			maxErr = err;
		}
	}
	//cout << "Hi 8 Exit ..." << endl;
}
//cout << "Hi 7 Exit ..." << endl;
//build the costliest overall path backward from its end via the table

ybar.setLength(x.getLength());
ybar.setTag(ybar.getLength() - 1, (tagID)maxI);
/*
if(y.getTag(t-1) != maxI)
	{
		maxErr = max(0, maxErr - 1);
	}
*/
//cout << "Hi 9 Enter ..." << endl;
for(int i = x.getLength() - 2; i > -1; i--)
{
	ybar.setTag(i, B[i+1][maxI][maxErr]);
	maxI = B[i+1][maxI][maxErr];
	if(y.getTag(i) != maxI)
	{
		maxErr = max(0, maxErr - 1);
	}
	
}
//cout << "Hi 9 Exit ..." << endl;

//	if(y == ybar) return label(); //special case: return empty label
return(ybar);
} // end function

int         empty_label(LABEL y)
{
	/* Returns true, if y is an empty label. An empty label might be
	   returned by find_most_violated_constraint_???(x, y, sm) if there
	   is no incorrect label that can be found for x, or if it is unable
	   to label x at all */
	return y.isEmpty();
}

/*
   add all entries from src into dest, where each entry is ID -> value
   (when an ID exists in dest already, its value will change)

   the return value should be freed using free_svector() after use
 */
inline SVECTOR* addFeatureVectors(const SVECTOR& v1, const SVECTOR& v2)
{
	return add_ss(&const_cast<SVECTOR&>(v1), &const_cast<SVECTOR&>(v2));
}

/*
   auxiliary to appendFeatureVectorWithFeatNumOffset():
   return the number of elements in the word list NOT COUNTING THE 0 that must come at the end
 */
unsigned int sparseVecLength(const SVECTOR& v)
{
	WORD* w = v.words;
	unsigned int count = 0;
	while(w->wnum != 0)
	{
		w++;
		count++;
	}
	return count;
}

/*
   stick src on the end of dest, adding offset to each feature number in src
 */
void appendFeatureVectorWithFeatNumOffset(SVECTOR& dest, const SVECTOR& src, unsigned int offset)
{
	unsigned int sizeDest = sparseVecLength(dest), sizeSrc = sparseVecLength(src);
	WORD* temp = dest.words;
	dest.words = (WORD*)my_malloc((sizeDest + sizeSrc + 1) * sizeof(WORD));
	memcpy(dest.words, temp, sizeDest * sizeof(WORD));
	free(temp); temp = NULL;
	for(WORD* w = dest.words + sizeDest, *w2 = src.words; w2->wnum != 0; w++, w2++)
	{
		w->wnum = w2->wnum + offset;
		w->weight = w2->weight;
	}
	dest.words[sizeDest + sizeSrc].wnum = 0;
}

SVECTOR     *psi(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
	/* Returns a feature vector describing the match between pattern x
	   and label y. The feature vector is returned as a list of
	   SVECTOR's. Each SVECTOR is in a sparse representation of pairs
	   <featurenumber:featurevalue>, where the last pair has
	   featurenumber 0 as a terminator. Featurenumbers start with 1 and
	   end with sizePsi. Featurenumbers that are not specified default
	   to value 0. As mentioned before, psi() actually returns a list of
	   SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
	   specifies the next element in the list, terminated by a NULL
	   pointer. The list can be thought of as a linear combination of
	   vectors, where each vector is weighted by its 'factor'. This
	   linear combination of feature vectors is multiplied with the
	   learned (kernelized) weight vector to score label y for pattern
	   x. Without kernels, there will be one weight in sm->w for each
	   feature. Note that psi has to match
	   find_most_violated_constraint_???(x, y, sm) and vice versa. In
	   particular, find_most_violated_constraint_???(x, y, sm) finds
	   the ybar!=y that maximizes psi(x,ybar,sm)*sm->w (where * is the
	   inner vector product) and the appropriate function of the
	   loss + margin/slack rescaling method. See that paper for details. */

	//for HMM purposes, there will be just one SVECTOR in the linked list, and the score for (x, y) is w * psi(x, y)
	SVECTOR *fvec = (SVECTOR*)my_malloc(sizeof(SVECTOR));
	fvec->factor = 1;
	fvec->userdefined = (char*)my_malloc(sizeof(char));	//leaving this uninitialized causes seg faults
	fvec->userdefined[0] = 0;								//(this value gets checked in create_svector() )
	fvec->next = NULL;

	/*
	   psi(x, y) contains a copy of each word x_i, offset depending on y_i, and a 1 earlier in the vector for each state->state transition
	 */

	//count state transitions and build a total feature vector for each tag that's used in sentence x
	hash_map<unsigned int, unsigned int> transitions; //one entry per tag->tag transition found in the input; the value is the count
	static SVECTOR** featuresByTag = new SVECTOR*[getNumTags()]; //tag ID -> map of feature IDs to sum of values for all words with said tag

	for(unsigned int i = 0; i < getNumTags(); i++)
	{
		featuresByTag[i] = (SVECTOR*)my_malloc(sizeof(SVECTOR));
		featuresByTag[i]->words = (WORD*)my_malloc(sizeof(WORD));
		featuresByTag[i]->words[0].wnum = 0;
		featuresByTag[i]->userdefined = NULL;
		featuresByTag[i]->next = NULL;
		featuresByTag[i]->factor = 1;
	}
	for(unsigned int i = 0; i < y.getLength() - 1; i++)
	{
		SVECTOR* tempVec = addFeatureVectors(*featuresByTag[y.getTag(i)], x.getToken(i).getFeatureMap());
		free_svector(featuresByTag[y.getTag(i)]);
		featuresByTag[y.getTag(i)] = tempVec;
		tempVec = NULL;
		transitions[get_transition_feature_id(y.getTag(i), y.getTag(i + 1))]++;
	}
	SVECTOR* tempVec = addFeatureVectors(*featuresByTag[y.getLastTag()], x.getLastToken().getFeatureMap());
	free_svector(featuresByTag[y.getLastTag()]);
	featuresByTag[y.getLastTag()] = tempVec;
	tempVec = NULL;

	fvec->words = (WORD*)my_malloc((transitions.size() + 1) * sizeof(WORD)); //allow space for the end-vector flag (feat. # 0)

	//add features to the vector in numerical order (transitions, then tag feature sums)
	unsigned int fvecIndex = 0; //index into output vector that we're currently writing

	//add the count of uses of each transition that's used
	for(unsigned int i = 0; i < getNumTags(); i++)
		for(unsigned int j = 0; j < getNumTags(); j++)
		{
			unsigned int id = get_transition_feature_id(i, j);
			if(transitions.find(id) != transitions.end())
			{
				fvec->words[fvecIndex].wnum = id; //feature numbers start at 1; this is handled in get_*_id()
				fvec->words[fvecIndex].weight = transitions[id];
				fvecIndex++;
			}
		}
	//add the end-of-list flag (that this is 0 is *why* feature numbers start at 1)
	fvec->words[fvecIndex].wnum = 0;

	//for each tag in order, add the sum of the feature vectors of the words so labeled
	for(unsigned int i = 0; i < getNumTags(); i++)
		if(featuresByTag[i]->words[0].wnum != 0) //there are tokens with this label
			appendFeatureVectorWithFeatNumOffset(*fvec, *featuresByTag[i], get_output_feature_start_id((tagID)i, sparm) - 1);

	//cleanup
	for(unsigned int i = 0; i < getNumTags(); i++)
		free_svector(featuresByTag[i]);

	return(fvec);
}

/*
   if the labels aren't the same length, the loss is computed using the appropriate subsequence of whichever label is longer
 */
double      loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm)
{
	/* loss for correct label y and predicted label ybar. The loss for
	   y==ybar has to be zero. sparm->loss_function is set with the -l option. */
	if(sparm->loss_function == 0)   /* type 0 loss: 0/1 loss */
		/* return 0, if y==ybar. return 1 else */
	{
		fprintf(stderr, "loss(): loss function is set to zero/one loss; this code only works with Hamming loss (loss_func = 1). exiting\n");
		exit(-1);
		const unsigned int minSize = min(y.getLength(), ybar.getLength());
		for(unsigned int i = 0; i < minSize; i++)
			if(ybar.getTag(i) != y.getTag(i))
				return 1;
		return 0;
	}
	/* Put your code for different loss functions here. But then
	   find_most_violated_constraint_???(x, y, sm) has to return the
	   highest scoring label with the largest loss. */
	else if(sparm->loss_function == 1) /* type 1 loss: constant penalty per wrong POS tag */
	{
		unsigned int penalty = 0;
		const unsigned int minSize = min(y.getLength(), ybar.getLength());
		for(unsigned int i = 0; i < minSize; i++)
			if(ybar.getTag(i) != y.getTag(i)) penalty++;
		return (double)penalty;
	}
	else
	{
		fprintf(stderr, "loss(): unknown loss function id %d\n", sparm->loss_function);
		exit(-1);
	}
}

int         finalize_iteration(double ceps, int cached_constraint,
		SAMPLE sample, STRUCTMODEL *sm,
		CONSTSET cset, double *alpha, 
		STRUCT_LEARN_PARM *sparm)
{
	/* This function is called just before the end of each cutting plane iteration. ceps is the amount by which the most violated constraint found in the current iteration was violated. cached_constraint is true if the added constraint was constructed from the cache. If the return value is FALSE, then the algorithm is allowed to terminate. If it is TRUE, the algorithm will keep iterating even if the desired precision sparm->epsilon is already reached. */
	return(0);
}

void        print_struct_learning_stats(SAMPLE sample, STRUCTMODEL *sm,
		CONSTSET cset, double *alpha,
		STRUCT_LEARN_PARM *sparm)
{
	/* This function is called after training and allows final touches to
	   the model sm. But primarily it allows computing and printing any
	   kind of statistic (e.g. training error) you might want. */
}

void        print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm,
		STRUCT_LEARN_PARM *sparm,
		STRUCT_TEST_STATS *teststats)
{
	/* This function is called after making all test predictions in
	   svm_struct_classify and allows computing and printing any kind of
	   evaluation (e.g. precision/recall) you might want. You can use
	   the function eval_prediction to accumulate the necessary
	   statistics for each prediction. */

	double avgLoss = (double)(teststats->numTokens - teststats->numCorrectTags) / teststats->numTokens;
	printf("average loss per word: %.4lf\n", avgLoss);
}

void        eval_prediction(long exnum, EXAMPLE ex, LABEL ypred,
		STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm,
		STRUCT_TEST_STATS *teststats)
{
	/* This function allows you to accumlate statistic for how well the
	   prediction matches the labeled example. It is called from
	   svm_struct_classify. See also the function
	   print_struct_testing_stats. */
	if(exnum == 0) /* this is the first time the function is called. So initialize the teststats (note it has been allocated) */
	{
		teststats->numTokens = teststats->numCorrectTags = 0;
	}
	teststats->numTokens += ex.x.getLength();
	for(unsigned int i = 0; i < ex.x.getLength(); i++)
		if(ex.y.getTag(i) == ypred.getTag(i))
			teststats->numCorrectTags++;
}

/*
   auxiliary to read/write_struct_model()
 */
string structModelFilename2svmModelFilename(const string& smFilename)
{
	return smFilename.substr(0, smFilename.rfind('.')) + "_svmModel.dat";
}

/*
   autogenerate a filename to which to write the svm model
 */
void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
	/* Writes structural model sm to file file. */

	ofstream outfile(file);
	//write number of features per word
	outfile << "feature space size: " << sparm->featureSpaceSize << endl;
	//write the tags we picked up from the input
	outfile << "labels:";
	for(hash_map<tagID, tag>::iterator i = idToTagMap.begin(); i != idToTagMap.end(); i++)
		outfile << " " << (*i).first << "=" << (*i).second;
	outfile << endl;
	//write the (sparse) weight vector
	outfile << "weight vector size: " << sm->sizePsi << endl;
	outfile << "weight vector:";
	for(unsigned int i = 0; i < (unsigned int)sm->sizePsi; i++)
		if(sm->w[i] != 0)
			outfile << " " << i << ":" << setprecision(8) << sm->w[i];
	outfile << endl;
	outfile << "loss type (1 = slack rescaling, 2 = margin rescaling): " << sparm->loss_type << endl;
	outfile << "loss function (should be 1 for svm-hmm): " << sparm->loss_function << endl;
	outfile.close();
	printf("writing svm model to '%s'\n", structModelFilename2svmModelFilename(file).c_str());
	write_model(const_cast<char*>(structModelFilename2svmModelFilename(file).c_str()), sm->svm_model); //write svm model
}

/*
   autogenerate a filename to check for the svm model
 */
STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm)
{
	/* Reads structural model sm from file file. This function is used
	   only in the prediction module, not in the learning module. */

	STRUCTMODEL model;

#define ERROR_READING(what) fprintf(stderr, "read_struct_model(): error reading " #what "\n"); exit(-1)

	ifstream infile(file);
	//read number of features per word
	if(!(infile >> match("feature space size: ") >> sparm->featureSpaceSize))
	{
		ERROR_READING("feature space size");
	}
	//read tags taken from input to learner
	if(!(infile >> match("\nlabels: ")))
	{
		ERROR_READING("labels");
	}
	string labelLine;
	if(!getline(infile, labelLine, '\n'))
	{
		ERROR_READING("labels");
	}
	//the model is read before the examples, so we can fill up the tag database without using the protective interface above
	unsigned int id;
	string label, idStr;
	istringstream inlbl(labelLine);
	while(getline(inlbl, idStr, '=') && inlbl >> label)
	{
		istringstream inid(idStr);
		if(!(inid >> id))
		{
			ERROR_READING("labels");
		}
		idToTagMap[id] = label;
		tagToIDMap[label] = id;
	}
	//read the (sparse) weight vector
	if(!(infile >> match("weight vector size: ") >> model.sizePsi))
	{
		ERROR_READING("weight vector size");
	}
	model.w = (double*)my_malloc(model.sizePsi * sizeof(double));
	memset(model.w, 0, model.sizePsi * sizeof(double)); //all entries default to 0
	if(!(infile >> match("\nweight vector: ")))
	{
		ERROR_READING("weight vector");
	}
	unsigned int featNum;
	double featVal;
	string featLine;
	if(!getline(infile, featLine, '\n'))
	{
		ERROR_READING("weight vector");
	}
	istringstream instr(featLine);
	while(instr >> featNum >> match(":") >> featVal)
		model.w[featNum] = featVal;
	//read the learning parameters
	if(!(infile >> match("loss type (1 = slack rescaling, 2 = margin rescaling): ") >> sparm->loss_type))
	{
		ERROR_READING("loss type");
	}
	if(!(infile >> match("\nloss function (should be 1 for svm-hmm): ") >> sparm->loss_function))
	{
		ERROR_READING("loss function");
	}

#undef ERROR_READING

	infile.close();
	setTagRegistryWritable(false); //make sure tags read in through the test set won't be used during classification
	model.svm_model = read_model(const_cast<char*>(structModelFilename2svmModelFilename(file).c_str())); //read svm model
	return model;
}

void        write_label(FILE *fp, LABEL y)
{
	/* Writes label y to file handle fp. Used only to output classification results. */
	fprintf(fp, "{ ");
	for(unsigned int i = 0; i < y.getLength(); i++)
		fprintf(fp, "%s ", getTagByID(y.getTag(i)).c_str());
	fprintf(fp, "}");
}

void        free_pattern(PATTERN x)
{
	/* Frees the memory of x. */
	//no-op
}

void        free_label(LABEL y) {
	/* Frees the memory of y. */
	//no-op
}

void        free_struct_model(STRUCTMODEL sm)
{
	/* Frees the memory of model. */
	/* if(sm.w) GC_FREE(sm.w); */ /* this is free'd in free_model */
	if(sm.svm_model) free_model(sm.svm_model, 1);
	/* add free calls for user defined data here */
}

void        free_struct_sample(SAMPLE s)
{
	/* Frees the memory of sample s. */
	//no-op; we don't know whether the examples were allocated via malloc() or new[]
}

void        print_struct_help()
{
	/* Prints a help text that is appended to the common help text of
	   svm_struct_learn. */
	printf("         --* string  -> custom parameters that can be adapted for struct\n");
	printf("                        learning. The * can be replaced by any character\n");
	printf("                        and there can be multiple options starting with --.\n");
}

void         parse_struct_parameters(STRUCT_LEARN_PARM *sparm)
{
	sparm->featureSpaceSize = 0; //this is checked when reading the examples

	/* Parses the command line parameters that start with -- */
	for(unsigned int i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
		switch ((sparm->custom_argv[i])[2])
		{
			case 'a': i++; /* strcpy(learn_parm->alphafile,argv[i]); */ break;
			case 'e': i++; /* sparm->epsilon=atof(sparm->custom_argv[i]); */ break;
			case 'k': i++; /* sparm->newconstretrain=atol(sparm->custom_argv[i]); */ break;
			default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]); exit(0);
		}
	}
}

void        print_struct_help_classify()
{
	/* Prints a help text that is appended to the common help text of
	   svm_struct_classify. */
	printf("         --* string -> custom parameters that can be adapted for struct\n");
	printf("                       learning. The * can be replaced by any character\n");
	printf("                       and there can be multiple options starting with --.\n");
}

void         parse_struct_parameters_classify(char *attribute, char *value)
{
	/* Parses one command line parameters that start with -- . The name
	   of the parameter is given in attribute, the value is given in
	   value. */

	switch (attribute[2]) 
	{ 
		/* case 'x': strcpy(xvalue,value); break; */
		default: printf("\nUnrecognized option %s!\n\n",attribute);
			 exit(0);
	}
}
