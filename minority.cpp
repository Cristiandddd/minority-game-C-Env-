/***************************************************************************
                          minority.cpp  -  description
                             -------------------
    begin                : 26 March 2023
	last                 : 8 April 2024
    email                : estevez@yahoo.com
 ***************************************************************************/
 
/***************************************************************************
 *   Copyright (C) 2024 by Ernesto Estevez Rams   						   *
 *   eerams@yahoo.com  											   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
 #include <vector>
 #include <iostream>
 #include <algorithm>
 #include <numeric>
 
 #include "configuration.h"
 #include "minority.h"

 #include <chrono>

//.............................................................................
//                      constructors
//.............................................................................

//.............................................................................
// Name: minority
//
// Sinopsis: constructor
//
// Parameters:
//           struct minority_parameters mino;
//
// Return: None
//        
// Exceptions:
//           BadAlloc()
//
// ............................................................................
minority::minority(struct minority_parameters& mino){
unsigned long P;
    
number_of_players=mino.number_of_players;
naive_players=( mino.naive < number_of_players)?  mino.naive : 0;
number_of_producers=(mino.producers < number_of_players)? mino.producers : 0;
number_of_strategies=mino.number_of_strategies;
initial_seed=mino.seed;
memory=mino.memory;


P=0x01<<memory;
alpha=static_cast<double>(P)/number_of_players;
	
if(mino.initial_mu==0){
  initial_mu=RNDInteger(P-1);
  mino.initial_mu=initial_mu;
	}
else
  initial_mu=mino.initial_mu;

Initialize();

}


minority::minority(struct minority_options& mino){
unsigned long P;
    
    
number_of_players=mino.number_of_players;
naive_players=( mino.naive < number_of_players)?  mino.naive : number_of_players;
number_of_producers=(mino.producers < number_of_players)? mino.producers : number_of_players;
number_of_strategies=mino.number_of_strategies;
memory=mino.memory;
initial_seed=mino.seed;
initial_agents=mino.initial_agents;
teq=mino.teq;
memory=mino.memory;

P=0x01<<memory;
alpha=static_cast<double>(P)/number_of_players;
teq=mino.teq*P;
	
if(mino.initial_mu==0){
  initial_mu=RNDInteger(P-1);
  mino.initial_mu=initial_mu;
	}
else
  initial_mu=mino.initial_mu;

Initialize();

}

minority::minority(const minority & mi){
	 *this=mi;
}

// ....................... End of constructors ...............................

minority & minority::operator=(const minority & mi){
		
    number_of_players=mi.number_of_players;;
    naive_players=mi.naive_players;
    number_of_producers=mi.number_of_producers;
	number_of_strategies=mi.number_of_strategies;
	teq=mi.teq;
	memory=mi.memory;
	initial_mu=mi.initial_mu;
	initial_seed=mi.initial_seed;
    initial_agents=mi.initial_agents;
	alpha=mi.alpha;
		
	players=mi.players;
	
	return *this;
}

minority & minority::operator=(struct minority_parameters& mino){
unsigned long P;
    
number_of_players=mino.number_of_players;
naive_players=( mino.naive < number_of_players)?  mino.naive : number_of_players;
number_of_producers=(mino.producers < number_of_players)? mino.producers : number_of_players;
number_of_strategies=mino.number_of_strategies;
memory=mino.memory;
initial_seed=mino.seed;
initial_agents=mino.initial_players;

P=0x01<<memory;
alpha=static_cast<double>(P)/number_of_players;
	
teq=mino.teq*P;
	
if(mino.initial_mu==0){
  initial_mu=RNDInteger(P-1);
  mino.initial_mu=initial_mu;
	}
else
  initial_mu=mino.initial_mu;

Initialize();

return *this;
}


//.............................................................................
// Name: Initialize
//
// Sinopsis: Resets the game
//
// Parameters:
//           None
//
// Return: None
//        
// Exceptions:
//           BadAlloc()
//
// ............................................................................
void minority::Initialize(void){
bool naiv=false;
unsigned long P=0x01<<memory;

	players.clear();
	
	/* initialisation of the players */
	for(int i=0; i< number_of_players; i++){
		if(i < naive_players)
		  naiv=true;
		else
		  naiv=false;
		  
		agent ag(i, P, number_of_strategies, naiv, false);
		players.push_back(ag);
		}
	 
	 // Setting producers randomly
	if(number_of_producers < number_of_players){
		int prod=0;
		int index=0;
		while(prod < number_of_producers){
			index=static_cast<int>(RNDInteger(number_of_players-1)); // set producers at random
			if(players[index].Producer()==false){
				players[index].Producer(true);
				prod++;
				}
			}
		}
	else { // all players are producers
		for(auto & pl : players)
			pl.Producer(true);
		}

}

//.............................................................................
// Name: Run
//
// Sinopsis: The banana !! Runs the minority game
//
// Parameters:
//           None
//
// Return: Number of players
//        
// Exceptions:
//           BadAlloc()
//
// ............................................................................ 

int minority::Run(void){
unsigned long mu=0;
unsigned long mu_naive=0;
// unsigned long mur=0;
// unsigned long mu_naiver=0;
int winBit=0;
unsigned long P=0x01<<memory;

// Initialize time
auto start = std::chrono::high_resolution_clock::now();

 /* beginning of the game */
mu_naive=mu=initial_mu; 
// mu_naiver=mur=initial_mu; 

// the banana !!
for(int round=0; round<number_of_players+teq+10000; round++){
	
    if(round==teq){ /* resets the measured quantities if the equilibration time is reached*/
		for(agent & ag : players){
			ag.ClearRecords();
			ag.SetStationary();
            ag.SetP(P);
			}
		}
        
    {
    int A=0; /* A(t) */
    
    for(std::vector<agent>::iterator it=players.begin(); it != players.end(); it++){ //betting. Everybody plays
        int b=it->Bet(mu, mu_naive);
        A+=b;
        }

	/* determining of the winning side */
	if(A > 0){
	  winBit=0;
	  }
	else{
	  winBit=1;
	  }

	// update scores only
    for (int i = 0; i < number_of_players; i++) {
        players[i].UpdateScore((players[i].Naive()) ? mu_naive : mu, A);
    }

	mu=(2*mu+winBit)%P; // real histories. 
	mu_naive=RNDInteger(P-1); //random histories 
    }
    
	}//for round

 // Finalize time
auto end = std::chrono::high_resolution_clock::now();

std::chrono::duration<double> duration = end - start;

std::cout<< "Time taken: " << duration.count() << " secs" << std::endl;

 return number_of_players;
 }
 

 //.............................................................................
// Name: RunBidirectional
//
// Sinopsis: The banana !! Runs the minority game. Bidirectional graph is built
//
// Parameters:
//           None
//
// Return: None
//        
// Exceptions:
//           BadAlloc()
//
// ............................................................................ 
int minority::RunBidirectional(void){
unsigned long mu=0;
unsigned long mu_naive=0;
int winBit=0;
int agents=initial_agents;
unsigned long P=0x01<<memory;
	
 /* beginning of the game */
mu_naive=mu=initial_mu;  

// the banana !!
for(int round=0; round<number_of_players+teq-initial_agents+1; round++){
	
    if(round==teq){ /* resets the measured quantities if the equilibration time is reached*/
		
		for(auto & ag : players){
			ag.ClearRecords();
			ag.SetStationary();
            ag.SetP(P);
			}
		}
    
    {
    int A=0; /* A(t) */
    
    for(std::vector<agent>::iterator it=players.begin(); it != players.end(); it++){
        int b=it->Bet(mu, mu_naive);
        A+=b;
        }

	/* determination of the winning side */
	if(A > 0){
	  winBit=0;
	  }
	else{
	  winBit=1;
	  }

	// update scores only
	// std::vector<agent>::iterator itagent=players.begin()+agents;
    int n=0;
	for(std::vector<agent>::iterator it=players.begin(); it!=players.end(); it++){
		it->UpdateScore((it->Naive())? mu_naive : mu, A);
		n++;
		}

	mu=(2*mu+winBit)%P; // real histories. 
	mu_naive=RNDInteger(P-1); //random histories 
    }
    
    if(round > teq)
        agents++;
	}
 
 return number_of_players;
 }
 
//.............................................................................
// Name: Clear
//
// Sinopsis: Clears all records keeping the game parameters
//
// Parameters:
//           None
//
// Return: None
//        
// Exceptions:
//           BadAlloc()
//
// ............................................................................
void minority::Clear(void) {

for(std::vector<agent>::iterator it=players.begin(); it!=players.end(); it++)
   it->ClearRecords();

}
