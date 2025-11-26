/***************************************************************************
                          agent.cpp  -  description
                             -------------------
    begin                : 26 March 2023
	last                 : 8 April 2024
    email                : eerams@yahoo.com
 ***************************************************************************/
 
/***************************************************************************
 *   Copyright (C) 2024 by Ernesto Estevez Rams   						   *
 *   eerams@yahoo.com											   *
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
#include "agent.h"


agent & agent::operator=(const agent & ag){

		  producer=ag.producer;
		  naive=ag.naive;
		  frozen=ag.frozen;
		  stationary=ag.stationary;
          id=ag.id;
		  
		  P=ag.P;
		  best_strategy=ag.best_strategy;
		  strategies=ag.strategies; // lookup table for strategies
	
return *this;
}

int agent::Initialize(int ide, unsigned long p, int number_of_strategies, bool naiv, bool prod){
	int it=0;
	
	producer=prod; 
	naive=naiv;
	frozen=true; 
	stationary=false;
	P=p;
    id=ide;
	strategies.clear();
	
	/* initialisation of the strategies */
  for(int j=0; j < number_of_strategies; j++){
		strategy str;
	  
		for(int mu=0; mu < P; mu++)
			str.look_up_table.push_back(2*RNDInteger(1)-1);
		
		for(auto str1 : strategies){
			   if(str1==str && it < MAXITERATIONSBEFOREGIVINGUP){
				 j--;
				 it++;
				 break;
				}
			}
			 
		strategies.push_back(str);
		it=0;
		}
    
    best_strategy=RNDInteger(number_of_strategies-1); // choose the best strategy randomly
    
 return number_of_strategies;
}

int agent::Bet(unsigned long mu, unsigned long mu_naive){
	 int bet=0;
	 int index=0;
	 
	 for(std::vector<strategy>::iterator it=strategies.begin(); it!=strategies.end(); it++){
		if(Producer()==false){
			if(strategies[best_strategy].score==it->score){
			  if(RNDDouble()<0.5) /* breaks ties */
				  best_strategy=index;
					}
			  else{
				if(strategies[best_strategy].score < it->score)
					best_strategy=index;
			   }
		   }else // if the player is a producer choose only first strategy
			  best_strategy=0;
		  
		index++;
		}
					
	 if(naive)
		bet=strategies[best_strategy].look_up_table[mu_naive];  // This player is naive
	else
		bet=strategies[best_strategy].look_up_table[mu];  // This player is not naive
		
	bet_record=bet;
	
		 	
	return bet;
}

bool agent::DidIWin(int win){
	return (bet_record*(2*win-1)>0);
}

void agent::UpdateScore(unsigned long mu, int A){

	  for(std::vector<strategy>::iterator it=strategies.begin(); it !=strategies.end(); it++)
		  it->score+=-it->look_up_table[mu]*A;
}
